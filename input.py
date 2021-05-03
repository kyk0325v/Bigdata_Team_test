# f = open("test1.txt", 'w')
#
# for i in range(1, 11):
#     data = "%d번째 줄입니다.\n" % i
#     f.write(data)
#
# f.close()

# f = open("hi.txt", 'r')
# line = f.readline()
# print(line)
# f.close()

# f = open("hi.txt", 'r')
# while True:
#     line = f.readline()
#     if not line: break
#     print(line)
# f.close()

# f = open("hi.txt", 'r')
# lines = f.readlines()
# for line in lines:
#     print(line)
# f.close()

# f = open("hi.txt", 'r')
# data = f.read()
# print(data)
# f.close()
#
# f = open("hi.txt",'a')
#
# for i in range(11, 20):
#     data = "%d번째 줄입니다.\n" % i
#     f.write(data)
#
# f.close()
#
# f = open("foo.txt", 'w')
# f.write("Life is too short, you need python")
# f.close()


# 절차지향언어 => 클래스 개념 X
# 객체지향언어 => 클래스 개념 O

# result = 0
#
# def add(num):
#     global result
#     result += num
#     return result
#
# print(add(3))
# print(add(4))
# print(add(10))
#
# result1 = 0
# result2 = 0
#
# def add1(num):
#     global result1
#     result1 += num
#     return result1
#
# def add2(num):
#     global result2
#     result2 += num
#     return result2
#
# print(add1(3))
# print(add1(4))
# print(add2(3))
# print(add2(7))

# class Calculator:   #2개의 메서드를 가지고 있다 => 2개의 기능이 있다.
#     def __init__(self):
#         self.result = 0
#
#     def add(self, num):
#         self.result += num
#         return self.result
#     def sub(self, num):
#         self.result -=num
#         return self.result
#
# cal1 = Calculator() #cal1이라는 객체를 생성했다.  #Calculator클래스의 인스턴스가 생성됐다.
# cal2 = Calculator() #cal2라는 객체를 생성했다.   #Calculator클래스의 인스턴스가 생성됐다.
# #객체 2개 cal1 cal2 Calculator()
# cal3 = Calculator()
# cal4 = Calculator()
# cal5 = Calculator()
#
# print("-----1------")
# print(cal1.add(3))
# print(cal1.add(4))
# print("------2-----")
# print(cal2.add(3))
# print(cal2.add(7))
#
# print("---5-------")
# print(cal5.add(10))
# print(cal5.add(10))
# print("----4------")
# print(cal4.add(100))
# print(cal4.add(100))

# class Cookie:
#
#     pass
#
#
#
#
# c1 = Cookie()   # c1은 cookie 클래스의 인스턴스이다. c1은 객체이다.
# c2 = Cookie()
# c3 = Cookie()
#
# class FourCal():
#     def add(self,num):
#         pass
#     def sub(self,num):
#         pass
#     def mul(self,num):
#         pass
#     def div(self,num):
#         pass
#     pass
# x=3000
# cal1 = FourCal()
# print(type(x))
# print(type(cal1))

# def add1(a,b):
#     return a+b
#
# aaa = add1(10,5)
# print(aaa)
# class FourCal:
#      def __init__(self, first, second):
#          self.first = first
#          self.second = second
#      def setdata(self, first, second):
#          self.first = first
#          self.second = second
#      def add(self):
#          result = self.first + self.second
#          return result
#      def mul(self):
#          result = self.first * self.second
#          return result
#      def sub(self):
#          result = self.first - self.second
#          return result
#      def div(self):
#          result = self.first / self.second
#          return result

# print(a.sub())
class Cal(): #기능이 5가지. 메서드 5개와 클래스변수 2개로 이루어져있다.
#     def __init__(self, first, second, mode):
#         self.first = first
#         self.second = second
    def setdata(self, first, second):
        self.first = first
        self.second = second
    def add(self):
        return self.first+self.second
    def sub(self):
        return self.first-self.second
    def mul(self):
        return self.first*self.second
    def div(self):
        return self.first/self.second

        # class UpGradeCal(Cal):  # 기능이 5가지는 상속을 받아왔고 1가지는 추가가 되었음. 6가지
        #     def pow(self):
        #         result = self.first ** self.second
        #         return result
        # MoreFourcal
#SafeCal  (Cal클래스를 상속받아왔음)
#1.setdata
# 2.add
# 3.sub
# 4.mul
# 5.div
#---------------------------------------
#1.def div
#       return 0
class safecal(Cal): #기능이 5가지 셋데이터 + - * /
     def div(self):
        if self.second == 0: # 나누는 값이 0인 경우 0을 리턴하도록 수정
            return 0
        else:
            return self.first / self.second

class Family:
     lastname = "김" # 클래스에 속한 변수.
     firstname = "철수" #클래스 변수.

f1 =Family()
f2 =Family()
# print(f1.lastname)
# print(f2.firstname)

# Family.lastname ="박"
# print(f1.lastname)

# print(Family.lastname,Family.firstname)

# a=Cal()
# safea=safecal()
# a.setdata(10,0)
# safea.setdata(10,0)
# print(a.div())
# print(safea.div())

# a = UpGradeCal()
# b = Cal()
# a.setdata(2, 10)
# print(a.pow())
# b.setdata(4, 0)
# print(b.div())

# a=SafeCal()
# a.setdata(10,3)
# print(a.div())
# #print(a.Div())
#
# b=Cal()
# b.setdata(10,3)
# print(b.div())
        # if mode== "add":
        #     print(self.first + self.second)
        # elif mode=="sub":
        #     print(self.first - self.second)
        # elif mode=="mul":
        #     print(self.first * self.second)
        # elif mode=="div":
        #     print(self.first / self.second)
        # else:
        #     print("없는 연산이다.")

# class MoreFourCal(FourCal):
#     pass
# a = MoreFourCal() # 기존클래스를 상속받은 클래스, 객체 생성.
# a.setdata(50,10) # 상속받은 클래스의 객체로 셋데이터 함수 호출.
# print(a.div()) # 상속받은 클래스의 객체로 div함수 호출.
# a = FourCal(30, 20, "add")

# print(a.cal("add"))
# print(a.cal("sub"))
# print(a.cal("mul"))
# print(a.cal("div"))


# a =FourCal(4,10) # 자동으로 객체가 생성됨과 동시에 생성자 메서드가 호출
# print(a.add())
# print(a.sub())
# print(a.mul())
# print(a.div())




#mycal=FourCal() #포칼 클래스 객체 mycal 생성
#mycal2=FourCal() #포칼 클래스 객체 mycal2 생성

#mycal.setdata(50,10)   #mycal 객체로 데이터셋 50,10
#mycal2.setdata(200,600) #mycal2 객체로 데이터셋 200,600
#result_mycal =mycal.add() #mycal 객체의 add메서드 호출
#result_mycal2 = mycal2.add() #mycal2 객체의 add메서드 호출

#print(result_mycal)
#print(result_mycal2)

         #self.x1 = m
         #self.x2 = m
         #self.x3 = m
         #self.x4 = n
         #self.x_sum = m+n
         #self.x_multi = m*n
         #self.x_sub = m-n
         #셋데이터 메서드
################################### 클래스
#xx= FourCal()
#xx.setdata(5,10,20,40) #호출 = 함수를 사용한다.
#yy.setdata(1,2,3,40)

#print(xx.x_multi)
#print(xx.x_sum)
#print(xx.x_sub)

#a=FourCal()
#b=FourCal()
#a.setdata(4,2)
#b.setdata(3,7)

#print(id(a.x1))
#print(id(a.x2))
#print(id(b.x1))
#print(id(b.x2))
#FourCal.setdata(a, 3 ,10) #호출하는 주체가 정해져있지 않다.
#FourCal.setdata(b, 5 ,8)
#a.setdata(3, 10) #호출하는 주체가 누군지 포함되어있다.
#print(a.x1)
#print(a.x2)
#print(b.x1)
#print(b.x2)


#def setdata():
     #return"111"
# 셋데이터 함수

#a = FourCal()
#b = FourCal()

#setdata
#setdata()           # 전역적 함수
#FourCal.setdata()   # 포칼 클래스의 함수를 함수를 호출
#a.setdata(5,10)         # 객체 a의 셋데이터 함수 호출
#b.setdata()         # 객체 b의 셋데이터 함수 호출
#print(a.x)
#print(a.y)

#print()
#type()
#int()
#str()


