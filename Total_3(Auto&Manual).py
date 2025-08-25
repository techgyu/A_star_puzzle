import heapq
import random
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyautogui
from matplotlib.widgets import Button


# swap 함수 정의
def swap(imgs, initial_state, i, blank_index):
    print("before imgs: ", imgs)
    imgs[i - 1], imgs[blank_index] = imgs[blank_index], imgs[i - 1]
    print("after imgs: ", imgs)
    # imgs가 변경되면 initial_state도 같이 변경
    row_i, col_i = (i - 1) // 3, (i - 1) % 3
    row_blank, col_blank = blank_index // 3, blank_index % 3
    print("before initial_state: ", initial_state)
    initial_state[row_i][col_i], initial_state[row_blank][col_blank] = initial_state[row_blank][col_blank], initial_state[row_i][col_i]
    print("after initial_state: ", initial_state)

def move_zero(grid, num_moves):
    # grid의 깊은 복사본 생성
    grid_copy = copy.deepcopy(grid)
    
    for _ in range(num_moves):
        # 0의 위치 찾기
        zero_row, zero_col = None, None
        for i in range(3):
            for j in range(3):
                if grid_copy[i][j] == '9':
                    zero_row, zero_col = i, j
                    break

        # 0을 상하좌우 중 랜덤하게 선택
        moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # 좌, 우, 상, 하
        random.shuffle(moves)
        for move in moves:
            new_row, new_col = zero_row + move[0], zero_col + move[1]
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                # 0과 주변 숫자 위치 바꾸기
                grid_copy[zero_row][zero_col], grid_copy[new_row][new_col] = grid_copy[new_row][new_col], grid_copy[zero_row][zero_col]
                break  # 한 번 이동 후 반복문 종료

    return grid_copy

# 목표 상태 설정
goal = [['1', '2', '3'], ['4', '5', '6'], ['7', '8', '9']]
# 각 숫자의 목표 위치; 저장
goal_positions = {value: (i, j) # value: (i, j) : 딕셔너리를 만드는데, 'value'를 키로 하고, 그 값이 위치한 좌표 (i, j)를 값으로 저장
                    for i ,row in enumerate(goal) # i: 행 인덱스, row: 해당 행의 전체 내용
                    for j, value in enumerate(row)} #j: 열 인덱스, value: 해당 행, 열의 값 
                    # enumerate(goal): 'goal'을 행 단위로 순회하면서 각 행과 그 행의 인덱스를 제공
                    # enumerate(row): 각행 'row'을 열 단위로 순회하면서 각 값과 그 값의 인덱스를 제공

# 현재 상태 설정
initial_state = move_zero(goal, 100)  # 9를 20번 랜덤하게 이동
print(initial_state)


#이미지 전달
imgList1 = ['reference/장원영/1.jpg', 'reference/장원영/2.jpg', 'reference/장원영/3.jpg', 'reference/장원영/4.jpg', 'reference/장원영/5.jpg', 'reference/장원영/6.jpg', 'reference/장원영/7.jpg', 'reference/장원영/8.jpg', 'reference/장원영/9.jpg']

# 이미지 이름에서 숫자를 추출하고 이를 이용해 매핑을 생성
img_dict = {img.split('/')[-1].split('.')[0]: img for img in imgList1}

# initial_state 배열을 이용해 imgList2를 생성
imgList2 = [img_dict[num] for row in initial_state for num in row]

print("imgList2 (initial_state 배열에 따른 이미지 리스트):")
print(imgList2)

###########################[Automate]################################

def print_grid(grid):
    print("----------------")
    for row in grid:
        print(" ".join(row))

# 깊이(gn) 계산: root 노드에서 얼만큼의 깊이를 갖고 있는 지 계산
def find_depth(node):
    depth = 0 # 깊이 0으로 설정
    while node.parent is not None: # parent가 없을 때까지,
        node = node.parent # 위로 올라감
        depth += 1
    return depth

# 휴리스틱 계산: Manhattan Distance를 사용
def heuristic_func(state):
    distance = 0
    for i in range(3):
        for j in range(3):
            value = state[i][j]
            if value != '9':
                goal_i, goal_j = goal_positions[value]
                distance += abs(goal_i - i) + abs(goal_j - j) # 절댓값 상으로 목표와 얼마나 떨어져 있는지 계산함
    return distance # 각각의 목표와 목표 지점까지의 거리를 모두 합산하여 반환

class GeneralTreeNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state # 데이터 값
        self.parent = parent # 부모 노드를 가리키는 포인터
        self.action = action # 현재 노드로 이동하기 위해 적용된 작업
        self.gn = find_depth(self) # 깊이 gn 계산
        self.heuristic = heuristic_func(state) # 휴리스틱 값 계산
        self.fn = self.gn + self.heuristic # fn 값 계산(경로 값 비교에 사용)
        self.children = [] # 자식 노드

    def add_child(self, child_node):
        self.children.append(child_node)

    def __lt__(self, other): # 클래스의 내용을 다른 객체와 비교
        return self.fn < other.fn # 자신의 fn 값을 다른 fn와 비교하여, 더 큰 것을 우선 반환

#인덱스 탐색
def find_index(matrix, value):
    for i, row in enumerate(matrix):
        if value in row:
            column = row.index(value)
            return (i, column)
    return None

#인접한 인덱스를 찾아 반환
def get_neighbors(matrix, position):
    
    r, c = position
    rows = len(matrix) #주어진 행렬(matrix)의 행(row) 수 계산: 항상 3으로 고정
    cols = len(matrix[0]) #주어진 행렬(matrix)의 열(col) 수 계산: 항상 3으로 고정
    adjacent_indices = []
    # 상, 하, 좌, 우 이웃 찾기
    if r > 0: #상 (가로로 2 ~ 3번째)
        adjacent_indices.append((r-1, c)) #바로 위 1번째 줄에 있는 인덱스 위치를 추가
    if r < rows - 1: #하 (가로로 1 ~ 2번째)
        adjacent_indices.append((r+1, c)) #바로 아래 1번째 줄에 있는 인덱스 위치를 추가
    if c > 0: #좌 (세로로 2 ~ 3번째)
        adjacent_indices.append((r, c-1))
    if c < cols - 1: #우 (세로로 1 ~ 2번째)
        adjacent_indices.append((r, c+1))
    return adjacent_indices

#새로운 행렬을 생성하고, 내용을 변경하는 연산
def swap_and_create_new_state(matrix, pos1, pos2):
    new_matrix = [row[:] for row in matrix] # 기존 행렬을 복사(리스트 컴프리헨션)하여 새로운 행렬 생성 - 원본 행렬을 변경하지 않음
    new_matrix[pos1[0]][pos1[1]], new_matrix[pos2[0]][pos2[1]] = new_matrix[pos2[0]][pos2[1]], new_matrix[pos1[0]][pos1[1]] # 두 위치의 값을 서로 교환 - 임시 변수 사용 안 함
    return new_matrix # 변경된 새로운 값을 반환

#A star 연산
def a_star_search(initial_state, goal):
    root = GeneralTreeNode(initial_state) #최초 값 initial_state로 root(최상위 노드 생성)
    open_list = [] # 열린 노드를 저장하는 리스트, 우선 순위 큐
    closed_set = set() # 닫힌 노드를 저장하는 set, 중복 방문 방지
    #heappush 연산: 새로운 요소를 힙의 맨 끝에 추가, 힙의 성질을 유지하는 위치로 이동[힙의 부모 노드와 반복 비교]
    heapq.heappush(open_list, root) #초기 상태가 들어있는 최상위 노드 root를 open_list에 추가함
    while open_list: # open_list가 비어 있지 않은 동안 반복한다
        #heappop 연산 : heap에서 최솟값을 삭제하고 반환하는 연산, 제거한 후에는 힙의 성질을 다시 유지함
        current_node = heapq.heappop(open_list) # currnet_node에 open_list에서 휴리스틱 값이 가장 작은 노드를 하나 꺼내서 할당(우선 순위가 높은)
        closed_set.add(tuple(tuple(row) for row in current_node.state)) #currnet_node.state(데이터)를 튜플로 변환하여 closed_node에 추가
        #각 행을 tuple로 변환하고: [('1', '2', '3'), ('4', '5', '6'), ('7', '8', '9')]
        #tuple을 갖고 있는 리스트 전체를 튜플로 변환함: (('1', '2', '3'), ('4', '5', '6'), ('7', '8', '9'))
        #tuple로 변경하는 이유: 집합(set) 자료 구조를 사용하려면 변경 불가능한 자료 구조여야 하므로

        #목표 달성 시
        if current_node.state == goal: # 만약 꺼내온 current_node.state가 goal과 같다면
            return current_node # 종료

        #목표 미 달성 시
        blank_index = find_index(current_node.state, '9') #9의 위치를 blank_index에 저장 
        neighbor_indices = get_neighbors(current_node.state, blank_index)#9와 인접해 있는 인덱스(n, n)을 찾아서 neighbor_indices에 저장
        for neighbor in neighbor_indices: # neighbor_indices를 하나씩 꺼내서 neighbor에 넣음
            new_state = swap_and_create_new_state(current_node.state, blank_index, neighbor) # blank_index와 neighbor의 위치가 교환된 새로운 state 생성
            if tuple(tuple(row) for row in new_state) not in closed_set: # closed_set에 있지 않은 상태인지 확인(중복 연산 방지)
                child_node = GeneralTreeNode(new_state, current_node, neighbor) # 새로운 상태가 맞다면, currnet_node의 자식으로 new_state를 추가하고, 이동한 방향 neighbor을 저장
                current_node.add_child(child_node)
                heapq.heappush(open_list, child_node) # 생성한 child_node를 open_list에 추가
    return None

def reconstruct_path(node):
    path = [] # 빈 리스트 path 선언
    while node.parent is not None: # 만약 매개변수 node의 parent가 공란이 아니라면
        path.append(node.action) # node의 action(이동 방향)을 path에 추가
        node = node.parent # 부모 노드로 이동
    return path[::-1] # path 리스트를 역순으로 반환하여 최초 시작점부터 마지막 끝점까지의 경로를 얻어냄

###########################[Automate]################################

#5inch * 5inch 도면 생성 / 3 by 3 서브플롯 생성 [fig: 전체 도면 / axes: 각 서브플롯]
fig, axes = plt.subplots(3, 3, figsize=(5, 5))
# 버튼 생성
ax_manual = plt.axes([0.78, 0.01, 0.1, 0.055])  # [left, bottom, width, height]
button_manual = Button(ax_manual, 'Manual')
ax_automatic = plt.axes([0.89, 0.01, 0.1, 0.055])  # [left, bottom, width, height]
button_automatic = Button(ax_automatic, 'Auto')

manual_automatic_plag = 1 # 최초 값 1(수동 모드)
# 수동 모드(manual_mode) 이벤트 핸들러 함수
def manual_mode(event):
    global manual_automatic_plag # 전역 변수 명시
    print("수동 모드 버튼이 클릭되었습니다.")
    # 여기에 수동 모드에 대한 동작 추가
    manual_automatic_plag = 1

# 자동 모드(automatic_mode) 이벤트 핸들러 함수
def automatic_mode(event):
    global manual_automatic_plag # 전역 변수 명시
    global initial_state # 전역 변수 명시
    print("자동 모드 버튼이 클릭되었습니다.")
    # 여기에 자동 모드에 대한 동작 추가
    manual_automatic_plag = 0

    solution_node = a_star_search(initial_state, goal) # initial_state에 대한 탐색 연산 실행

    if solution_node: # 만약 solution_node 결과가 나왔다면
        solution_path = reconstruct_path(solution_node) #path 경로를 재 설정 해주고
        print("Solution path:", solution_path) #solution path를 출력하여 보인다
    else:
        print("No solution found")

    print("[Goal]")
    print_grid(goal)
    print("[initial_state]")
    print_grid(initial_state)
    for path in solution_path:
        blank_index = find_index(initial_state, '9')
        initial_state = swap_and_create_new_state(initial_state, blank_index, path)
        print_grid(initial_state)

        # 이미지 배열 동기화
        imgs = [cv2.cvtColor(cv2.imread(img_dict[num]), cv2.COLOR_BGR2RGB) for row in initial_state for num in row]

        # 이미지 서브플롯에 표시
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.axis('off')
            plt.imshow(imgs[i])

        plt.pause(2)
        plt.draw()

    plt.clf()
    plt.imshow(completed_imgs)
    plt.axis('off')
    plt.show()
    

# 버튼 이벤트 연결
button_manual.on_clicked(manual_mode)
button_automatic.on_clicked(automatic_mode)

#cv2.imread(경로) : OpenCV 라이브러리에서 제공하는 이미지 파일을 읽어오는 함수
#imgN : cv2.imread(경로) 반환하는 값을 저장하는 변수, 이미지 데이터를 포함하는 다차원 배열(numpy array)
#cv2.cvtColor(이미지 파일(numpy array), 원하는 이미지 색상 변환)
imgs = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in imgList2]
sorted_imgs = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in imgList1]
completed_imgs = cv2.cvtColor(cv2.imread('main.jpg'), cv2.COLOR_BGR2RGB)
# 이미지 서브플롯에 표시
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.axis('off') #축을 숨김
    plt.imshow(imgs[i])

#두 점 사이의 거리가 특정 값 이하인 경우, 인접으로 간주
def neiborhood(subplot_num, blank_index):
    # 각 인덱스에 대한 인접한 인덱스를 딕셔너리로 정의
    nearby_indexes = {
        1: [2, 4],
        2: [1, 3, 5],
        3: [2, 6],
        4: [1, 5, 7],
        5: [2, 4, 6, 8],
        6: [3, 5, 9],
        7: [4, 8],
        8: [5, 7, 9],
        9: [6, 8]
    }
    print("subplot_num: ", subplot_num)
    print("blank_index: ", blank_index)
    print("is it neiborhood?: ", subplot_num in nearby_indexes[blank_index])
    return subplot_num in nearby_indexes[blank_index]

# 공백 이미지에 대한 데이터 생성(def add_point 안에서 쓰임)
img0 = cv2.cvtColor(cv2.imread('9.jpg'), cv2.COLOR_BGR2RGB)
#add_point 이벤트 핸들러 함수, 수동 모드일 때만 작동하도록 변경
def add_point(event):
    global manual_automatic_plag # 전역 변수 명시
    if event.button ==1 and manual_automatic_plag == 1: # 마우스 왼쪽 버튼을 클릭했고, 수동 모드 상태라면
        # 좌표 처리
        fore = pyautogui.getActiveWindow() # 현재 창을 확인
        pos = pyautogui.position() # 마우스 클릭이 발생한 위치를 확인
        x = pos.x - fore.left # 현재 마우스 커서의  x좌표 - 활성 창의 왼쪽 가장 자리 좌표
        print("pos.x: ", pos.x, " fore.left: ", fore.left)
        y = pos.y - fore.top # 현재 마우스 커서의 y 좌표 - 활성 창의 상단 가장 자리 좌표
        print("Mouse   : ", x, ", ", y) # 좌표 출력

        # 현재 공백 이미지가 들어가 있는 서브 플롯의 위치를 확인
        for i, img_data in enumerate(imgs): # 순회문
            if np.array_equal(img_data, img0):  # img0과 이미지 데이터가 동일한지 확인
                print("공백 인덱스의 위치. 인덱스:", i)
                blank_index = i
                break
        
        x1 = 70
        x2 = 198
        x3 = 201
        x4 = 328
        x5 = 331
        x6 = 457
        
        y1 = 92
        y2 = 217
        y3 = 220
        y4 = 347
        y5 = 350
        y6 = 475

        if (x >= x1 and x <= x2) and (y >= y1 and y <= y2):
            i = 1
            print("subplot ", i, "번")
            if(neiborhood(i, blank_index + 1) == True):
                # 공백 이미지와 비공백 이미지 교체
                swap(imgs, initial_state, i, blank_index)
            
                #2번 서브플롯 업데이트: 공백
                plt.subplot(3, 3, i)
                plt.axis('off') #축을 숨김
                plt.imshow(imgs[i - 1])
                #원래 공백이 있던 서브플롯 업데이트
                plt.subplot(3, 3, blank_index + 1)
                plt.axis('off') #축을 숨김
                plt.imshow(imgs[blank_index])

        if (x >= x3 and x <= x4) and (y >= y1 and y <= y2):
            i = 2
            print("subplot ", i, "번")
            if(neiborhood(i, blank_index + 1) == True):
                # 공백 이미지와 비공백 이미지 교체
                swap(imgs, initial_state, i, blank_index)
            
                #2번 서브플롯 업데이트: 공백
                plt.subplot(3, 3, i)
                plt.axis('off') #축을 숨김
                plt.imshow(imgs[i - 1])
                #원래 공백이 있던 서브플롯 업데이트
                plt.subplot(3, 3, blank_index + 1)
                plt.axis('off') #축을 숨김
                plt.imshow(imgs[blank_index])

        if (x >= x5 and x <= x6) and (y >= y1 and y <= y2):
            i = 3
            print("subplot ", i, "번")
            if(neiborhood(i, blank_index + 1) == True):
                # 공백 이미지와 비공백 이미지 교체
                swap(imgs, initial_state, i, blank_index)
            
                #2번 서브플롯 업데이트: 공백
                plt.subplot(3, 3, i)
                plt.axis('off') #축을 숨김
                plt.imshow(imgs[i - 1])
                #원래 공백이 있던 서브플롯 업데이트
                plt.subplot(3, 3, blank_index + 1)
                plt.axis('off') #축을 숨김
                plt.imshow(imgs[blank_index])

        if (x >= x1 and x <= x2) and (y >= y3 and y <= y4):
            i = 4
            print("subplot ", i, "번")
            if(neiborhood(i, blank_index + 1) == True):
                # 공백 이미지와 비공백 이미지 교체
                swap(imgs, initial_state, i, blank_index)
            
                #2번 서브플롯 업데이트: 공백
                plt.subplot(3, 3, i)
                plt.axis('off') #축을 숨김
                plt.imshow(imgs[i - 1])
                #원래 공백이 있던 서브플롯 업데이트
                plt.subplot(3, 3, blank_index + 1)
                plt.axis('off') #축을 숨김
                plt.imshow(imgs[blank_index])

        if (x >= x3 and x <= x4) and (y >= y3 and y <= y4):
            i = 5
            print("subplot ", i, "번")
            if(neiborhood(i, blank_index + 1) == True):
                # 공백 이미지와 비공백 이미지 교체
                swap(imgs, initial_state, i, blank_index)
            
                #2번 서브플롯 업데이트: 공백
                plt.subplot(3, 3, i)
                plt.axis('off') #축을 숨김
                plt.imshow(imgs[i - 1])
                #원래 공백이 있던 서브플롯 업데이트
                plt.subplot(3, 3, blank_index + 1)
                plt.axis('off') #축을 숨김
                plt.imshow(imgs[blank_index])

        if (x >= x5 and x <= x6) and (y >= y3 and y <= y4):
            i = 6
            print("subplot ", i, "번")
            if(neiborhood(i, blank_index + 1) == True):
                # 공백 이미지와 비공백 이미지 교체
                swap(imgs, initial_state, i, blank_index)
            
                #2번 서브플롯 업데이트: 공백
                plt.subplot(3, 3, i)
                plt.axis('off') #축을 숨김
                plt.imshow(imgs[i - 1])
                #원래 공백이 있던 서브플롯 업데이트
                plt.subplot(3, 3, blank_index + 1)
                plt.axis('off') #축을 숨김
                plt.imshow(imgs[blank_index])

        if (x >= x1 and x <= x2) and (y >= y5 and y <= y6):
            i = 7
            print("subplot ", i, "번")
            if(neiborhood(i, blank_index + 1) == True):
                # 공백 이미지와 비공백 이미지 교체
                swap(imgs, initial_state, i, blank_index)
            
                #2번 서브플롯 업데이트: 공백
                plt.subplot(3, 3, i)
                plt.axis('off') #축을 숨김
                plt.imshow(imgs[i - 1])
                #원래 공백이 있던 서브플롯 업데이트
                plt.subplot(3, 3, blank_index + 1)
                plt.axis('off') #축을 숨김
                plt.imshow(imgs[blank_index])

        if (x >= x3 and x <= x4) and (y >= y5 and y <= y6):
            i = 8
            print("subplot ", i, "번")
            if(neiborhood(i, blank_index + 1) == True):
                # 공백 이미지와 비공백 이미지 교체
                swap(imgs, initial_state, i, blank_index)
            
                #2번 서브플롯 업데이트: 공백
                plt.subplot(3, 3, i)
                plt.axis('off') #축을 숨김
                plt.imshow(imgs[i - 1])
                #원래 공백이 있던 서브플롯 업데이트
                plt.subplot(3, 3, blank_index + 1)
                plt.axis('off') #축을 숨김
                plt.imshow(imgs[blank_index])

        if (x >= x5 and x <= x6) and (y >= y5 and y <= y6):
            i = 9
            print("subplot ", i, "번")
            if(neiborhood(i, blank_index + 1) == True):
                # 공백 이미지와 비공백 이미지 교체
                swap(imgs, initial_state, i, blank_index)
            
                #2번 서브플롯 업데이트: 공백
                plt.subplot(3, 3, i)
                plt.axis('off') #축을 숨김
                plt.imshow(imgs[i - 1])
                #원래 공백이 있던 서브플롯 업데이트
                plt.subplot(3, 3, blank_index + 1)
                plt.axis('off') #축을 숨김
                plt.imshow(imgs[blank_index])

        # 이미지 정렬 여부 확인
        i = 0
        for img1, img2 in zip (imgs, sorted_imgs): #imgs의 인덱스 img1, sorted_imgs의 인덱스 img2
            if not np.array_equal(img1, img2): #비교해서 둘이 같지 않으면 멈춤
                break
            else: #이미지가 모두 정렬된 상황
                i = i + 1 # 카운터

        if(i == 9): # 정렬된 이미지가 9 개가 되면
                plt.clf() # 현재 figure를 지우고
                plt.imshow(completed_imgs) #완성 이미지 출력
                plt.axis('off') #축을 숨김
        
        plt.show()

cid = plt.connect('button_press_event', add_point)
plt.subplots_adjust(wspace=0.01, hspace=0.02)
plt.show()