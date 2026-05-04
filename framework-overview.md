```Python
import numpy as np

def run_flairr_ts(X_train, X_recent, actual_y, initial_prompt, H, L, M=2, N_iter=5, tau_stop=0.05):
    """
    FLAIRR-TS 알고리즘 파이썬 구현체 (Algorithm 1 기준)
    
    X_train: 과거 전체 데이터베이스 (검색용)
    X_recent: 현재 시점의 입력 데이터 윈도우 (크기 L)
    actual_y: MAE 계산을 위한 실제 정답 (크기 H)
    H: 예측 길이 (Horizon)
    L: 문맥 길이 (Context length)
    """
    
    # 1. 초기화
    p_curr = initial_prompt
    p_best = initial_prompt
    mae_min = float('inf')
    x_hat_best = None
    early_stop = False
    
    # 프롬프트와 오차 기록을 저장할 히스토리 리스트
    history = [] 

    # 2. 과거 유사 구간 검색 및 문맥 증강 (검색 에이전트 역할)
    # 현재 데이터(X_recent)와 가장 유사한 상위 M(2)개의 과거 구간을 찾음
    s_retr = retrieve_segments(X_train, X_recent, M) 
    c_aug = augment_context(X_recent, s_retr) # 쉼표로 구분된 텍스트로 변환

    # 3. 반복 튜닝 루프 시작 (Iterative Refinement)
    for k in range(N_iter):
        print(f"--- Iteration {k+1} ---")
        
        # [예측 에이전트] 현재 프롬프트와 증강된 문맥으로 LLM API 호출 및 텍스트 파싱
        x_hat_cand = forecaster_llm(p_curr, c_aug, H)
        
        # 결과에 대한 MAE 오차 계산
        mae_curr = calculate_mae(x_hat_cand, actual_y)
        print(f"Current MAE: {mae_curr}")
        
        # 최고 성능 업데이트
        if mae_curr < mae_min:
            mae_min = mae_curr
            p_best = p_curr
            x_hat_best = x_hat_cand
            
        # 기록 저장 (정제 에이전트에게 제공하기 위함)
        history.append({"prompt": p_curr, "mae": mae_curr})
        
        # [정제 에이전트] 전체 기록을 바탕으로 다음 프롬프트 수정 및 조기 종료 판단
        p_next, done_signal = refiner_llm(history, p_curr, tau_stop)
        
        # 조기 종료 조건 만족 시 (예: MAE 개선율이 5% 미만일 때)
        if done_signal:
            print("Stopping criteria met. Early stopping...")
            p_out = p_curr
            early_stop = True
            break
            
        # 다음 루프를 위해 프롬프트 업데이트
        p_curr = p_next

    # 4. 최대 반복 도달 후 종료 시 가장 좋았던 프롬프트 선택
    if not early_stop:
        print("Max iterations reached. Using the best prompt...")
        p_out = p_best
        
    return p_out, x_hat_best

# ==========================================
# (참고) 각 에이전트 및 유틸리티 함수의 가상 구현
# ==========================================
def retrieve_segments(history_db, current_window, M):
    # 피어슨 상관계수를 사용해 history_db에서 current_window와 가장 유사한 M개 추출
    pass

def augment_context(current_window, retrieved_segments):
    # 검색된 구간과 정답을 {raft_context} 포맷의 쉼표 구분 문자열로 텍스트화
    pass

def forecaster_llm(prompt, context, horizon):
    # 실제로는 openai.ChatCompletion.create() 나 genai.generate_content() 호출
    # 출력된 텍스트에서 'Predicted Values: [...]' 부분을 정규표현식(Regex)으로 파싱하여 배열(list)로 반환
    pass

def calculate_mae(predicted, actual):
    # np.mean(np.abs(np.array(predicted) - np.array(actual)))
    pass

def refiner_llm(history_log, current_prompt, tau_stop):
    # LLM을 호출하여 'Learnings: ...' 피드백을 받고 새 프롬프트(p_next)를 합성함
    # 'Done: True/False' 텍스트를 파싱하여 boolean 값(done_signal) 반환
    # 내부적으로 이전 MAE와 현재 MAE의 개선율을 계산하여 tau_stop(0.05)과 비교
    return next_prompt_string, boolean_done_signal
```