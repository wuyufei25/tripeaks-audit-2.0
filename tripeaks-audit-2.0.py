import streamlit as st
import pandas as pd
import numpy as np
import chardet
import io

# 1. 页面基础配置
st.set_page_config(page_title="Tripeaks 审计平台2.0", layout="wide")
st.title("🎴 Tripeaks 算法对比与深度审计平台2.0")

# --- 【工具函数：严防 NameError】 ---
def get_col_safe(df, target_keywords):
    for col in df.columns:
        c_str = str(col).replace(" ", "").replace("\n", "")
        for key in target_keywords:
            if key in c_str: return col
    return None

def calculate_advanced_stats(series, trim_percentage):
    """底层统计引擎：保持 15% 截断统计逻辑不变"""
    if len(series) < 5: 
        m = series.mean(); v = series.var()
        return m, v, (np.sqrt(v)/m if m > 0 else 0)
    sorted_s = np.sort(series)
    n = len(sorted_s)
    trim = int(n * (trim_percentage / 100))
    trimmed = sorted_s[trim : n - trim] if trim > 0 else sorted_s
    mu, var = np.mean(trimmed), np.var(trimmed)
    cv = (np.sqrt(var) / mu) if mu > 0 else 0
    return mu, var, cv

# --- 【修改点1：在函数参数中增加 collapse_thr】 ---
def audit_engine(row, col_map, base_init_score, burst_window, burst_threshold, collapse_thr):
    """审计引擎：保留原有逻辑，新增得分构成记录"""
    try:
        seq_raw = str(row[col_map['seq']])
        seq = [int(x.strip()) for x in seq_raw.split(',') if x.strip() != ""]
        desk_init = row[col_map['desk']]
        diff = row[col_map['diff']]
        actual = str(row[col_map['act']])
    except: 
        return 0, "解析失败", 0, 0, 0, 0, 0, 0, "数据错误", 0, 0, 0

    score = base_init_score
    breakdown = [f"基础分({base_init_score})"] 
    
    # --- 辅助统计数据 (用于Excel导出) ---
    max_combo = max(seq) if seq else 0
    long_combo_cnt = sum(1 for x in seq if x >= 3)
    valid_hand_cnt = sum(1 for x in seq if x > 0)
    
    # A. 基础体验分
    eff_idx = [i for i, x in enumerate(seq) if x >= 3]
    
    if sum(seq[:3]) >= 4: 
        score += 5; breakdown.append("开局破冰(+5)")
        
    if any(x >= 3 for x in seq[-5:]): 
        score += 5; breakdown.append("尾部收割(+5)")
        
    if len(seq) >= 7 and max(seq) in seq[6:]: 
        score += 5; breakdown.append("逆风翻盘(+5)")
    
    relay = 0
    if len(eff_idx) >= 2:
        for i in range(len(eff_idx)-1):
            if (eff_idx[i+1]-eff_idx[i]-1) <= 1: relay += 1
    
    # --- 接力分数保持 3, 5, 7 ---
    relay_score = (7 if relay >= 3 else 5 if relay == 2 else 3 if relay == 1 else 0)
    # ----------------------------------
    
    score += relay_score
    if relay_score > 0: breakdown.append(f"连击接力(+{relay_score})")

    # B. 贫瘠区扣分
    c1, c2, c3 = 0, 0, 0
    boundaries = [-1] + eff_idx + [len(seq)]
    for j in range(len(boundaries)-1):
        start, end = boundaries[j]+1, boundaries[j+1]
        inter = seq[start:end]
        if inter:
            L, Z = len(inter), inter.count(0)
            if L >= 6 or (L >= 4 and Z >= 3): 
                c3 += 1; score -= 25 if start <= 2 else 20
                breakdown.append(f"枯竭区(-{'25' if start <= 2 else '20'})")
            elif L == 5 or (3 <= L <= 4 and Z == 2): 
                c2 += 1; score -= 9
                breakdown.append("阻塞区(-9)")
            elif L >= 3: 
                c1 += 1; score -= 5
                breakdown.append("平庸区(-5)")

    # C. 自动化局判定
    f1, f2, red_auto = 0, 0, False
    con_list = []
    cur = 0
    for x in seq:
        if x > 0: cur += 1
        else:
            if cur > 0: con_list.append(cur); cur = 0
    if cur > 0: con_list.append(cur)
    
    for fl in con_list:
        if fl >= 7: red_auto = True
        elif 5 <= fl <= 6: 
            f2 += 1; score -= 9
            breakdown.append("过度投喂(-9)")
        elif fl == 4: 
            f1 += 1; score -= 3
            breakdown.append("高频投喂(-3)")

    # D. 红线判定
    red_tags = []
    
    # --- 【修改点2：将固定的 0.4 改为由滑块传参控制的比例】 ---
    if max(seq) >= desk_init * (collapse_thr / 100.0): red_tags.append("数值崩坏")
    
    if red_auto: red_tags.append("自动化局")
    if (diff <= 30 and "失败" in actual) or (diff >= 40 and "胜利" in actual): red_tags.append("逻辑违逆")
    
    total_eliminated = sum(seq)
    if total_eliminated > 0 and len(seq) >= burst_window:
        is_burst = False
        for i in range(len(seq) - burst_window + 1):
            if sum(seq[i : i + burst_window]) / total_eliminated >= (burst_threshold / 100):
                is_burst = True; break
        if is_burst: red_tags.append("消除高度集中")
    
    return score, ",".join(red_tags) if red_tags else "通过", c1, c2, c3, relay, f1, f2, " | ".join(breakdown), max_combo, long_combo_cnt, valid_hand_cnt

# --- 2. 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 审计全局参数")
    base_score = st.slider("审计初始分 (Base)", 0, 100, 65)
    mu_limit = st.slider("及格门槛 (μ)", 0, 100, 50)
    # --- 红线率控制滑块 ---
    red_rate_limit = st.slider("红线率容忍度 (%)", 0, 100, 15)
    # -------------------------
    
    st.divider()
    st.subheader("⚠️ 节奏风控红线")
    
    # --- 【修改点3：侧边栏新增控制滑块，初始值为 40】 ---
    collapse_thr = st.slider("数值崩坏阈值 (%)", 0, 100, 40)
    
    burst_win = st.number_input("连续手牌数 (窗口大小)", 1, 10, 3)
    burst_thr = st.slider("消除占比阈值 (%)", 0, 100, 80)
    st.divider()
    trim_val = st.slider("截断比例 (%)", 0, 30, 15)
    cv_limit = st.slider("最大 CV (稳定性)", 0.05, 0.50, 0.20)
    var_limit = st.slider("最大方差保护", 10, 100, 40)
    uploaded_files = st.file_uploader("📂 上传测试数据", type=["xlsx", "csv"], accept_multiple_files=True)

# --- 3. 计算流程 ---
if uploaded_files:
    raw_list = []
    for f in uploaded_files:
        try:
            if f.name.endswith('.xlsx'): t_df = pd.read_excel(f)
            else:
                raw_b = f.read(); enc = chardet.detect(raw_b)['encoding'] or 'utf-8'
                t_df = pd.read_csv(io.BytesIO(raw_b), encoding=enc)
            t_df['__ORIGIN__'] = f.name 
            raw_list.append(t_df)
        except Exception as e: st.error(f"读取 {f.name} 错误: {e}")

    if raw_list:
        main_df = pd.concat(raw_list, ignore_index=True)
        # 扩展列映射：精准定位“剩余桌面牌盖压关系”和“测试轮次”
        cm = {
            'seq': get_col_safe(main_df, ['全部连击']), 
            'desk': get_col_safe(main_df, ['初始桌面牌']),
            'diff': get_col_safe(main_df, ['难度']), 
            'act': get_col_safe(main_df, ['实际结果']),
            'hand': get_col_safe(main_df, ['手牌数量']), 
            'jid': get_col_safe(main_df, ['解集ID']),
            'rem_hand': get_col_safe(main_df, ['剩余手牌']), 
            'rem_desk_num': get_col_safe(main_df, ['剩余桌面牌', '剩余桌面']), # 纯数字计数
            'rem_desk_detail': get_col_safe(main_df, ['剩余桌面牌盖压关系']),   # 具体的点数花色
            'round_idx': get_col_safe(main_df, ['测试轮次', '轮次'])         # 原文件中的轮次
        }

        with st.spinner('执行红线并集概率审计...'):
            # --- 【修改点4：在 apply 调用时传入 collapse_thr】 ---
            audit_res = main_df.apply(lambda r: pd.Series(audit_engine(r, cm, base_score, burst_win, burst_thr, collapse_thr)), axis=1)
            
            main_df[['得分', '红线判定', 'c1', 'c2', 'c3', '接力', 'f1', 'f2', '得分构成', '最长连击', '长连次数', '有效手牌']] = audit_res

            fact_list = []
            for (f_n, h_v, j_i, d_v), gp in main_df.groupby(['__ORIGIN__', cm['hand'], cm['jid'], cm['diff']]):
                total = len(gp)
                is_break = gp['红线判定'].str.contains("数值崩坏")
                is_auto  = gp['红线判定'].str.contains("自动化局")
                is_logic = gp['红线判定'].str.contains("逻辑违逆")
                is_burst = gp['红线判定'].str.contains("消除高度集中")
                
                is_any_red = is_break | is_auto | is_logic | is_burst
                total_red_rate = is_any_red.sum() / total
                
                mu, var, cv = calculate_advanced_stats(gp['得分'], trim_val)
                reason = "✅ 通过"
                
                # --- 使用滑块控制的红线率阈值 ---
                if total_red_rate >= (red_rate_limit / 100):
                    mode_reason = gp[is_any_red]['红线判定'].str.split(',').explode().mode()[0]
                    reason = f"❌ 红线拒绝 ({mode_reason})"
                elif mu < mu_limit: reason = "❌ 分值拒绝"
                elif cv > cv_limit: reason = "❌ 稳定性拒绝"
                elif var > var_limit: reason = "❌ 波动拒绝"
                
                fact_list.append({
                    "源文件": f_n, "初始手牌": h_v, "解集ID": j_i, "难度": d_v,
                    "μ_均值": mu, "σ²_方差": var, "CV_变异系数": cv, # <--- 新增 CV 数据
                    "判定结论": reason,
                    "总红线率": total_red_rate, "数值崩坏率": is_break.mean(),
                    "自动化率": is_auto.mean(), "逻辑违逆率": is_logic.mean(), "爆发集中率": is_burst.mean(),
                    "is_pass": 1 if "✅" in reason else 0
                })
            df_fact = pd.DataFrame(fact_list)

        # === 4.1 看板展示 (全局平均分) ===
        st.header("📊 算法策略看板")
        strat_rows = []
        for h_v, gp_h in df_fact.groupby('初始手牌'):
            # 1. 计算通过数量 (Pass Count)
            pass_subset = gp_h[gp_h['is_pass'] == 1]
            diff_pass_cnt = pass_subset.groupby('难度').size().to_dict()
            
            # 2. 计算【全局】平均分 (Global Average Score)
            diff_global_avg = gp_h.groupby('难度')['μ_均值'].mean().to_dict()
            
            total_pass_jid = pass_subset.drop_duplicates(subset=['源文件', '解集ID']).shape[0]
            total_unique_jid = gp_h.drop_duplicates(subset=['源文件', '解集ID']).shape[0]
            
            row = {
                "手牌数": h_v, 
                "牌集总数": total_unique_jid, 
                "✅ 通过(去重)": total_pass_jid, 
                "覆盖率": total_pass_jid/total_unique_jid if total_unique_jid>0 else 0
            }
            
            # 填充难度列
            for d in sorted(df_fact['难度'].unique()):
                cnt = diff_pass_cnt.get(d, 0) # 通过的数量
                avg = diff_global_avg.get(d, 0) # 全局的均分
                if avg > 0 or cnt > 0:
                    row[f"难度{d} (通过|均分)"] = f"{cnt} (μ={avg:.1f})"
                else:
                    row[f"难度{d} (通过|均分)"] = "0"
            
            strat_rows.append(row)
        st.dataframe(pd.DataFrame(strat_rows).style.format({"覆盖率":"{:.1%}"}), use_container_width=True)

        # === 4.2 牌集风险明细排行 (新增 CV 展示) ===
        st.divider()
        st.subheader("🎯 牌集风险明细排行 (并集概率校验)")
        f_h = st.multiselect("手牌维度", sorted(df_fact['初始手牌'].unique()), default=sorted(df_fact['初始手牌'].unique()))
        f_s = st.radio("判定过滤", ["全部", "通过", "拒绝"], horizontal=True)

        view_df = df_fact[df_fact['初始手牌'].isin(f_h)].copy()
        if f_s == "通过": view_df = view_df[view_df['is_pass'] == 1]
        elif f_s == "拒绝": view_df = view_df[view_df['is_pass'] == 0]

        st.dataframe(view_df.drop(columns=['is_pass']).style.applymap(
            lambda x: 'color: #ff4b4b' if '❌' in str(x) else 'color: #008000', subset=['判定结论']
        ).format({
            "μ_均值":"{:.2f}", "σ²_方差":"{:.2f}", "CV_变异系数":"{:.2f}", # <--- 新增格式化
            "总红线率":"{:.1%}", 
            "数值崩坏率":"{:.1%}", "自动化率":"{:.1%}", "逻辑违逆率":"{:.1%}", "爆发集中率":"{:.1%}"
        }), use_container_width=True)
        st.info(f"📊 数据核查：当前列表共有 {len(view_df[view_df['is_pass']==1])} 行通过记录，看板与明细已完全对齐。")

        # === 4.3 新增：Excel 下载模块 (保留所有修复) ===
        with st.sidebar:
            st.divider()
            st.header("📥 导出审计详情")
            export_df = main_df.copy()
            
            # 构建输出列映射
            export_cols = {
                '__ORIGIN__': '关卡ID',
                cm['jid']: '解集ID',
                cm['round_idx']: '测试轮次',   
                cm['diff']: '难度',
                cm['act']: '实际结果',
                cm['rem_hand']: '剩余手牌',
                cm['rem_desk_num']: '剩余桌面牌数',      
                cm['rem_desk_detail']: '剩余桌面牌详情', 
                '最长连击': '最长连击',
                '长连次数': '长连次数',
                cm['seq']: '全部连击',
                '有效手牌': '有效手牌',
                cm['desk']: '初始桌面牌',
                cm['hand']: '初始手牌',
                '得分': '得分',
                '红线判定': '红线判定',
                '得分构成': '得分构成'
            }
            
            # 1. 仅重命名存在的列，缺失列补 N/A
            final_export_cols = {}
            for k, v in export_cols.items():
                if k is not None and k in export_df.columns:
                    final_export_cols[k] = v
                elif v in ['剩余手牌', '剩余桌面牌数', '剩余桌面牌详情', '测试轮次']: 
                    if k is None: export_df[v] = 'N/A' 
                    else: final_export_cols[k] = v 

            export_df = export_df.rename(columns=final_export_cols)

            # 2. 恢复轮次
            if '测试轮次' not in export_df.columns:
                 export_df.insert(2, '测试轮次', range(1, 1 + len(export_df)))
            
            # 3. 筛选最终输出列
            target_cols = ['关卡ID', '解集ID', '测试轮次', '难度', '实际结果', 
                           '剩余手牌', '剩余桌面牌数', '剩余桌面牌详情', 
                           '最长连击', '长连次数', '全部连击', '有效手牌', '初始桌面牌', '初始手牌', 
                           '得分', '红线判定', '得分构成']
            
            target_cols = [c for c in target_cols if c in export_df.columns]
            
            csv_data = export_df[target_cols].to_csv(index=False).encode('utf-8-sig')
            
            st.download_button(
                label="📄 下载完整审计明细 (Excel)",
                data=csv_data,
                file_name="Tripeaks_Audit_Details.csv",
                mime="text/csv"
            )
