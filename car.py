class Car:

    def __init__(self, _name='Avante', _efficiency=10, _maxtank=50):
        self.name = _name
            self.fe = _efficiency
        self.max = _maxtank
        self.fuel = 0  # 남은 연로량
        self.mileage = 0  # 누적 주행거리

    # 연료 충전
    def chargeFuel(self, fuel):
        before = self.fuel
        if fuel + self.fuel > self.max:
            self.fuel = self.max
        else:
            self.fuel += fuel
        print(f'[Fuel Charged : {self.fuel - before}]\n')

    # 차량 주행
    def drive(self, dist):
        before = self.mileage
        if dist > self.drivingDistance():
            self.fuel = 0
            self.mileage += self.drivingDistance()
        else:
            self.fuel -= dist / self.fe
            self.mileage += dist
        print(f'[Driving : {self.mileage - before}]\n')

    # 주행가능 거리
    def drivingDistance(self):
        return self.fuel * self.fe

    # 정보 출력
    def print(self):
        print('-' * 50)
        print(f'<Car Name, {self.name}>')
        fuel_info = f'Remaining Fuel\t: {self.fuel}/{self.max}L\nFuel Efficiency\t: {self.fe} Km/liter'
        dist_info = f'Max Driving Dis\t: {self.drivingDistance()} Km'
        mileage_info = f'Current Mileage\t: {self.mileage} Km'

        print(fuel_info)
        print(dist_info)
        print(mileage_info)
        print('-' * 50)
        print()
