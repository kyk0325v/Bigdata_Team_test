# 1. 사칙연산 기능이 있는 클래스를 만들고.
# 2. 그 클래스로 3개의 계산기 객체를 만들고.
# 3. 각각의 계산기가 독립적으로 작동되도록.
# 4. 계산기 3개가 총 몇번 구동되었는지 확인.
# 5. 계산 2개의 숫자를 입력해서 2개씩 계산하도록.
class Cal():
    total=0#클래스 뱐수 계산기가 총 몇번 돌아갔나. 체크
    def setdata(self, x, y):
        self.x = x
        self.y = y
    def add(self):
        self.result = self.x + self.y
        return self.resutl
    def sub(self):
        self.result = self.x - self.y
        return self.resutl
    def mul(self):
        self.result = self.x * self.y
        return self.resutl
    def div(self):
        self.result = self.x / self.y
        return self.resutl
    else:
        print("업데이트가 안되었습니다.")

cal1 = Cal()
cal2 = Cal()
cla3 = Cal()

cal1.result=0
cal2.result=0
cal3.result=0