# 1. 출석프로그램
# 2. 입장 ID 입력
# 3. ID에 맞는 개인정보가 저장
# 4. 출석체크
# 5. 체크시간이 기록
# 6. 조회 현황
# 7. 퇴실 체크
# 8. 조회 현황
#
#
# -->datetime
# -->datetime 가장 먼저 온 사람 ~ 늦게 온 사람 정렬
# --> 지각 기준 시간
# --> 지각자 조회
# --> 함수를 만들고 클래스르 만들 것
import datetime

list_id = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"]
list_member = []
list_time = []
userinput = int(input("1.입실 2.퇴실 3.조회\n"))

while True:
    if userinput ==1:
        print("입실")
        userID = input("ID: ")
        #ID입력받아서 ID list에 유효한 아이디인지 검토
        if userID in list_id:
            list_member.append(userID)
            list_time.append(datetime)
            print("출석 시간은 :",datetime.datetime.now())
            print("출석한 사람은 :", list_member)

            # 유효한 아이디
            # 유효하면 출석체크 시간 datetime으로 추가
            # 출석자 리스트에 출석자 이름 추가
            # 출석자 시간 리스트에 출석자 출석 시간 정보 추가
        else:
            print("명부에 없는 인원")
            #유효하지 않으면 '명부에 없는 인원' 메시지 출력

    elif userinput==2:#퇴실
        print("퇴실-ID입력")
        #퇴실 ID가 유효한지 확인
        #유효할 경우 현재 참여인원에서 해당 인원 제거
        #유효하지 않을 경우 '해당 ID 존재하ㅎ지 않는다' 출력
    elif userinput==3:#조회
        print("현황 조회")
        #참여 인원 목록 출력
        #인원별 출석 시간 출력
        #퇴실자 출력
        #퇴실 시간 출려
    else: #1,2,3이 아닌 입력 경우
        print("X")




