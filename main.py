from car import Car

c = Car('마이카', 15, 60)  # 이름, 연비, 연료탱크량 # Car 클래스 객체 c 생성, (차량이름, 연비, 최대연료탱크량)
c.print()  # Car 클래스 print() 메서드 호출, 차량 정보 출력

c.chargeFuel(47) #Car 클래스 chargeFuel() 메서드 호출, 기름 충전
c.print()

c.drive(100) #Car 클래스 drive() 메서드 호출, 차량 주행
c.print()
