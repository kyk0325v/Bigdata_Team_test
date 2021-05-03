main.py
import calculators.grade_calculator
import calculators.normal_calculator
from calculators import obesity_calculator
from calculators.pay_calculator import calculate_pay
from games.beskin_rabins_game import validate_input


if __name__ == "__main__":

    while True:

        print('''
실행하고자 하는 번호를 입력해주세요.
0. 종료
1. 학점 계산기
2. 일반 계산기
3. 비만도 계산기
4. 급여 계산기
        ''')

        choice = validate_input("선택 : ", ['0', '1', '2', '3', '4'])

        if choice == 1:
            calculators.grade_calculator.calculate_grade()
        elif choice == 2:
            calculators.normal_calculator.calculate_two_numbers()
        elif choice == 3:
            obesity_calculator.calculate_obesity()
        elif choice == 4:
            calculate_pay()
        elif choice == 0:
            print("프로그램을 종료합니다.")
            break