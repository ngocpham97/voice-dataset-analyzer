from collections import defaultdict


class L1Classifier:
    def __init__(self):
        self.keywords = {
            "customer_service": [
                "tài khoản", "hỗ trợ", "khiếu nại", "đơn hàng", "tổng đài", "phí", "xác minh", "OTP"
            ],
            "news": [
                "bản tin", "ngày", "giờ", "địa danh", "tên cơ quan"
            ],
            "education": [
                "khái niệm", "ví dụ", "nguyên lý", "bài học", "chương", "mục"
            ],
            "business": [
                "agenda", "kế hoạch", "OKR", "KPI", "ngân sách", "timeline", "phòng ban"
            ],
            "podcast": [
                "trải nghiệm", "suy ngẫm"
            ],
            "entertainment": [
                "trò chuyện", "gameshow", "showbiz"
            ],
            "technical": [
                "cài đặt", "lỗi", "debug"
            ],
            "finance": [
                "lãi suất", "lạm phát", "cổ phiếu", "tăng trưởng", "doanh thu"
            ],
            "health": [
                "triệu chứng", "chẩn đoán", "điều trị", "thuốc", "bác sĩ"
            ],
            "legal": [
                "điều khoản", "hợp đồng", "nghị định", "khiếu kiện"
            ],
            "children": [
                "học", "đọc", "thiếu nhi"
            ],
            "religion": [
                "kinh", "nghi lễ", "phong tục"
            ],
            "sports": [
                "tỷ số", "trận đấu", "cầu thủ"
            ]
        }

    def classify(self, transcript):
        scores = defaultdict(int)

        for category, words in self.keywords.items():
            for word in words:
                if word in transcript:
                    scores[category] += 1

        if not scores:
            return "unknown"

        return max(scores, key=scores.get)

    def get_keywords(self, category):
        return self.keywords.get(category, [])

    def add_keywords(self, category, new_keywords):
        if category in self.keywords:
            self.keywords[category].extend(new_keywords)
        else:
            self.keywords[category] = new_keywords

    def remove_keywords(self, category, keywords_to_remove):
        if category in self.keywords:
            self.keywords[category] = [word for word in self.keywords[category] if word not in keywords_to_remove]
