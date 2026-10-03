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

## 3. Phân Tích Đánh Giá Trade-Off (Lợi - Hại)

### 3.1. Khả Năng Ghi Nhớ Xuyên Phiên (Cross-Session Recall)
- **Baseline Agent**: Điểm recall chạm mốc **0.0%** trong bài kiểm tra stress test dài hạn. Lý do vì Baseline khóa bộ nhớ theo `thread_id`; khi các câu hỏi recall được gửi ở một thread riêng biệt, agent hoàn toàn không có thông tin.
- **Advanced Agent**: Đạt điểm recall **25.0%** trong stress test và **13.9%** trong standard benchmark. Điểm số này vượt trội nhờ tầng **Persistent `User.md`** cho phép agent đọc lại các fact đã trích xuất từ trước dù đang ở phiên làm việc mới.

### 3.2. Hiệu Quả Token & Giảm Tải Prompt (Prompt Load Reduction)
- Ở bộ dữ liệu ngắn (Standard), Baseline tiêu tốn ít prompt token hơn vì không phải mang theo nội dung file `User.md` và bản tóm tắt.
- Tuy nhiên, trong **Long-Context Stress Benchmark** (hội thoại kéo dài 16+ lượt):
  - **Baseline**: Prompt token tăng liên tục theo chiều dài hội thoại (**37,234 tokens**) do phải giữ lại toàn bộ lịch sử thô.
  - **Advanced**: Nhờ kích hoạt **28 lần compaction** để thu gọn các đoạn hội thoại cũ thành summary, tổng prompt token đã giảm xuống còn **34,734 tokens** (tiết kiệm ~2,500 prompt tokens), đồng thời số token phản hồi tự sinh cũng giảm từ **2,398** xuống **1,652**.

### 3.3. Đánh Đổi Về Dung Lượng Lưu Trữ & Độ Phức Tạp
- **Dung lượng đĩa (Memory Growth)**: Advanced Agent tốn thêm trung bình **201 - 214 bytes** cho mỗi người dùng để lưu trữ file `User.md`. Đây là chi phí lưu trữ cực kỳ nhỏ so với lợi ích ghi nhớ lâu dài mang lại.
- **Độ phức tạp tính toán**: Advanced Agent đòi hỏi nhiều logic xử lý hơn (Regex fact extraction, File I/O, Compaction check), nhưng đổi lại giữ được Response Quality cao gấp đôi (40.0% so với 20.0%).

---

## 4. Kết Luận

1. **Short-term memory duy nhất là không đủ** cho các ứng dụng AI Agent thực tế vì agent sẽ hoàn toàn quên bối cảnh người dùng ngay khi khởi tạo phiên làm việc mới.
2. **Persistent Memory (`User.md`)** là chìa khóa giúp giải quyết bài toán Cross-Session Recall với chi phí lưu trữ tối ưu.
3. **Compact Memory (Compacting/Summarization)** giúp kiểm soát hiện tượng phình đại ngữ cảnh (Prompt Explosion) ở các hội thoại dài, giữ chi phí token ổn định và tối ưu hóa thời gian phản hồi.
