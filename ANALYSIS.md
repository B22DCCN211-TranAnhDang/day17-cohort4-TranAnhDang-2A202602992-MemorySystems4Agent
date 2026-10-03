# Báo Cáo Phân Tích Hệ Thống Bộ Nhớ Agent (Memory Systems Benchmark Analysis)

## 1. Tổng Quan Kiến Trúc Bộ Nhớ

Trong bài lab này, chúng ta đã xây dựng và so sánh hai kiến trúc bộ nhớ cho AI Agent:

1. **Baseline Agent (Agent A)**:
   - **Cơ chế**: Chỉ sử dụng Short-term Memory theo phiên (`SessionState`), được lưu trữ tạm thời trong RAM và khóa theo `thread_id`.
   - **Đặc điểm**: Khi chuyển sang phiên/thread mới, toàn bộ lịch sử hội thoại cũ bị xóa hoàn toàn. Không có bộ nhớ bền vững (`User.md`) và không thực hiện nén dữ liệu (`compactions = 0`).

2. **Advanced Agent (Agent B)**:
   - **Cơ chế**: Kết hợp 3 tầng bộ nhớ mở rộng:
     1. **Within-session memory**: Lưu trữ tin nhắn lượt thoại gần đây.
     2. **Persistent Memory (`User.md`)**: Trích xuất các thực thể/thông tin cố định (tên, nơi ở, nghề nghiệp, sở thích) và lưu trữ bền vững xuống ổ đĩa thông qua `UserProfileStore`.
     3. **Compact Memory (`CompactMemoryManager`)**: Tự động giám sát ngưỡng token. Khi chuỗi hội thoại quá dài, hệ thống thực hiện nén (summarization) các tin nhắn cũ và chỉ giữ lại các tin nhắn mới nhất.

---

## 2. Kết Quả Benchmark Chi Tiết

Sau khi thực thi `python src/benchmark.py`, hai bộ dữ liệu thử nghiệm cho ra kết quả thực nghiệm như sau:

### 📊 Bảng 1: Standard Benchmark (`data/conversations.json`)

| Agent | Agent Tokens Only | Prompt Tokens Processed | Recall Score | Response Quality | Memory Growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 2,242 | 17,688 | 1.8% | 21.4% | 0 | 0 |
| **Advanced** | 2,757 | 23,849 | **13.9%** | **31.1%** | 201 | 11 |

### 🚀 Bảng 2: Long-Context Stress Benchmark (`data/advanced_long_context.json`)

| Agent | Agent Tokens Only | Prompt Tokens Processed | Recall Score | Response Quality | Memory Growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 2,398 | 37,234 | **0.0%** | 20.0% | 0 | 0 |
| **Advanced** | 1,652 | **34,734** | **25.0%** | **40.0%** | 214 | **28** |

---

## 3. Phân Tích Chi Tiết Theo 4 Câu Hỏi Trọng Tâm (Bước 8 Guide.md)

### 3.1. Vì sao Advanced Agent có Recall vượt trội hơn Baseline Agent?
- **Số liệu chứng minh**: Trong bài kiểm tra Long-Context Stress Benchmark, Cross-Session Recall của Advanced Agent đạt **25.0%** (và **13.9%** ở Standard Benchmark), trong khi Baseline Agent sụt giảm về **0.0%**.
- **Cơ chế trong code**: Khi người dùng cung cấp thông tin, `extract_profile_updates()` trích xuất các fact cố định và ghi xuống đĩa thông qua `UserProfileStore` (`User.md`). Ở thread/session mới, `_offline_response()` của Advanced Agent nạp lại `User.md` để trả lời chính xác câu hỏi recall, trong khi `BaselineAgent` khóa bộ nhớ theo `thread_id` nên quên hoàn toàn dữ liệu ở phiên mới.
- **Giới hạn**: Khả năng recall của Advanced Agent phụ thuộc vào độ chính xác của bộ trích xuất fact (`extract_profile_updates`); nếu tin nhắn thô dùng từ ngữ không khớp với mẫu trích xuất, fact có thể bị bỏ sót.

### 3.2. Vì sao Advanced Agent lại tốn nhiều Prompt Tokens hơn ở hội thoại ngắn?
- **Số liệu chứng minh**: Ở Standard Benchmark (hội thoại ngắn ~10 lượt), tổng **Prompt Tokens Processed** của Advanced Agent là **23,849 tokens**, cao hơn mức **17,688 tokens** của Baseline Agent.
- **Cơ chế trong code**: Trong `_estimate_prompt_context_tokens()`, mỗi lượt thoại của Advanced Agent bắt buộc phải load thêm nội dung file `User.md` và bản tóm tắt `summary` vào ngữ cảnh prompt.
- **Giới hạn**: Ở các hội thoại ngắn, số lượng lượt thoại chưa đủ nhiều để cơ chế compaction phát huy tác dụng nén lịch sử, dẫn đến khoản overhead cố định từ `User.md` làm chi phí prompt token cao hơn.

### 3.3. Vì sao Compact Memory đem lại lợi thế vượt trội ở hội thoại dài?
- **Số liệu chứng minh**: Trong Long-Context Stress Benchmark (hội thoại dài 16+ lượt kèm dữ liệu nhiễu), tổng **Prompt Tokens Processed** của Advanced Agent giảm xuống còn **34,734 tokens** (thấp hơn mức **37,234 tokens** của Baseline), đồng thời ghi nhận **28 lần Compactions**.
- **Cơ chế trong code**: Khi tổng token của thread vượt ngưỡng `compact_threshold_tokens`, `CompactMemoryManager.append()` kích hoạt `summarize_messages()` để cô đọng lịch sử cũ thành bản tóm tắt ngắn, loại bỏ các tin nhắn rác cồng kềnh.
- **Giới hạn**: Compact memory chủ yếu tối ưu hóa cho cột **Prompt Tokens Processed** bằng cách cắt giảm độ dài bối cảnh; nó không làm giảm kích thước của các fact cố định trong `User.md`.

### 3.4. Sự tăng trưởng file memory và các rủi ro hệ thống đi kèm
- **Số liệu chứng minh**: **Memory Growth (bytes)** của Advanced Agent tăng trung bình **201 - 214 bytes** per user, trong khi Baseline Agent luôn giữ mốc **0 bytes**.
- **Cơ chế trong code**: `UserProfileStore.write_text()` ghi các fact dưới dạng cấu trúc Markdown (`state/profiles/{user_id}.md`). Hàm `_update_user_profile()` thực hiện ghi đè (overwrite) khi nhận được fact mới cho cùng một key để tránh nhân đôi dữ liệu.
- **Rủi ro quan sát**: Nếu người dùng đưa ra các thông tin sai hoặc nhiễu, hệ thống có thể trích xuất nhầm và lưu lâu dài vào `User.md`. Ngoài ra, khi dữ liệu người dùng tích lũy hàng năm, số lượng fact trong `User.md` có thể phình to, làm tăng bối cảnh prompt ban đầu.

---

## 4. Hướng Mở Rộng Nâng Cấp (Bonus Extension - Target Mốc Điểm 90-100)

Để hướng tới mốc điểm tối đa **90-100** theo `Rubric.md`, chúng ta lựa chọn hướng mở rộng: **Conflict Handling & Confidence Thresholding cho Persistent Memory**.

### 4.1. Vấn đề thực tế giải quyết
Trong quá trình thử nghiệm, nếu người dùng đặt câu hỏi nghi vấn (ví dụ: *"Mình sống ở Hà Nội phải không?"*), các bộ trích xuất đơn giản dễ nhầm lẫn câu hỏi thành một fact mới và ghi đè dữ liệu cũ. Ngoài ra, khi người dùng thay đổi thông tin (Correction), hệ thống cần xử lý xung đột thông tin cũ/mới một cách thông minh.

### 4.2. Tác động tới Recall và Token Cost
- **Về Recall**: Thêm bộ lọc `Confidence Threshold` giúp loại bỏ hoàn toàn các thực thể nhiễu từ câu hỏi, tăng độ chính xác (`Precision`) của thông tin lưu trong `User.md`.
- **Về Token Cost**: Giúp dung lượng file `User.md` luôn gọn gàng, giảm bớt lượng prompt token thừa phải mang theo ở mỗi lượt thoại.

### 4.3. Đánh đổi & Rủi ro hệ thống (System Complexity & Trade-offs)
- **Độ phức tạp**: Cần bổ sung thêm một mô hình Classifier/LLM Judge nhỏ để chấm điểm độ tin cậy (`confidence score > 0.8`) trước khi cho phép ghi vào `User.md`.
- **Rủi ro**: Nếu thiết lập ngưỡng `confidence` quá khắt khe, hệ thống có thể vô tình bỏ qua các fact quan trọng do người dùng diễn đạt bằng câu nói tự nhiên hoặc ẩn ý.

---

## 5. Kết Luận

1. **Short-term memory đơn thuần** hoàn toàn không đáp ứng được yêu cầu ghi nhớ lâu dài ở các ứng dụng AI Agent thực tế (Recall = 0.0% ở phiên mới).
2. **Persistent Memory (`User.md`)** là giải pháp bắt buộc để giải bài toán Cross-Session Recall.
3. **Compact Memory (`CompactMemoryManager`)** là chìa khóa kiểm soát hiện tượng bùng nổ prompt token ở các chuỗi hội thoại dài.
