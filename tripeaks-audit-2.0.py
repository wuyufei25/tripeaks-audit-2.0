import streamlit as st
import pandas as pd
import chardet
import io

# 1. 页面基础配置
st.set_page_config(page_title="逻辑违逆独立审计工具", layout="wide")
st.title("🎴 Tripeaks 逻辑违逆独立审计平台")

# --- 【工具函数】 ---
def get_col_safe(df, target_keywords):
    for col in df.columns:
        c_str = str(col).replace(" ", "").replace("\n", "")
        for key in target_keywords:
            if key in c_str: return col
    return None

def check_logic_violation(row, col_map):
    """
    独立逻辑违逆判定引擎
    胜测(10-30)：如果实际结果包含"失败"，则判定违逆（30为险胜）
    败测(40-60)：如果实际结果包含"胜利"，则判定违逆
    """
    try:
        diff = row[col_map['diff']]
        actual = str(row[col_map['act']])
        try: num_diff = int(float(diff))
        except: num_diff = 0
    except: 
        return False, "解析失败"

    # 修正后的双轨判定逻辑
    if num_diff <= 30 and "失败" in actual:
        return True, "低/中难败局"
    elif num_diff >= 40 and "胜利" in actual:
        return True, "高难爽局"
        
    return False, "正常"

# --- 2. 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 审计全局参数")
    
    # 独立的逻辑违逆容忍度滑块
    logic_rate_limit = st.slider("逻辑违逆容忍度 (%)", 0, 100, 15) 
    
    st.divider()
    uploaded_files = st.file_uploader("📂 上传测试数据", type=["xlsx", "csv"], accept_multiple_files=True)

# --- 3. 计算流程 ---
if uploaded_files:
    raw_list = []
    for f in uploaded_files:
        try:
            if f.name.endswith('.xlsx'): 
                t_df = pd.read_excel(f)
            else:
                raw_b = f.read()
                enc = chardet.detect(raw_b)['encoding'] or 'utf-8'
                t_df = pd.read_csv(io.BytesIO(raw_b), encoding=enc)
            t_df['__ORIGIN__'] = f.name 
            raw_list.append(t_df)
        except Exception as e: 
            st.error(f"读取 {f.name} 错误: {e}")

    if raw_list:
        main_df = pd.concat(raw_list, ignore_index=True)
        cm = {
            'diff': get_col_safe(main_df, ['难度']), 
            'act': get_col_safe(main_df, ['实际结果']),
            'jid': get_col_safe(main_df, ['解集ID']),
            'hand': get_col_safe(main_df, ['手牌数量', '初始手牌']),
            'round_idx': get_col_safe(main_df, ['测试轮次', '轮次'])         
        }

        with st.spinner('执行逻辑违逆独立审计...'):
            # 应用违逆判定
            audit_res = main_df.apply(lambda r: pd.Series(check_logic_violation(r, cm)), axis=1)
            main_df[['是否违逆', '违逆类型']] = audit_res

            fact_list = []
            # 按文件、初始手牌、解集ID、难度分组统计
            for (f_n, h_v, j_i, d_v), gp in main_df.groupby(['__ORIGIN__', cm['hand'], cm['jid'], cm['diff']]):
                total = len(gp)
                is_logic = gp['是否违逆'] == True
                
                # 计算逻辑违逆率
                logic_rate = is_logic.sum() / total if total > 0 else 0
                
                reason = "✅ 通过"
                if logic_rate >= (logic_rate_limit / 100):
                    reason = "❌ 逻辑违逆拒绝"
                
                fact_list.append({
                    "源文件": f_n, 
                    "初始手牌": h_v, 
                    "解集ID": j_i, 
                    "难度": d_v,
                    "测试样本数": total,
                    "违逆次数": is_logic.sum(),
                    "逻辑违逆率": logic_rate, 
                    "判定结论": reason,
                    "is_pass": 1 if "✅" in reason else 0
                })
            df_fact = pd.DataFrame(fact_list)

        # === 4.1 看板展示 ===
        st.header("📊 逻辑违逆风险明细")
        
        f_h = st.multiselect("手牌维度过滤", sorted(df_fact['初始手牌'].unique()), default=sorted(df_fact['初始手牌'].unique()))
        f_s = st.radio("判定过滤", ["全部", "通过", "拒绝"], horizontal=True)

        view_df = df_fact[df_fact['初始手牌'].isin(f_h)].copy()
        if f_s == "通过": view_df = view_df[view_df['is_pass'] == 1]
        elif f_s == "拒绝": view_df = view_df[view_df['is_pass'] == 0]

        st.dataframe(view_df.drop(columns=['is_pass']).style.map(
            lambda x: 'color: #ff4b4b' if '❌' in str(x) else 'color: #008000', subset=['判定结论']
        ).format({
            "逻辑违逆率":"{:.1%}"
        }), use_container_width=True)
        
        pass_count = len(view_df[view_df['判定结论'] == '✅ 通过'])
        fail_count = len(view_df[view_df['判定结论'] == '❌ 逻辑违逆拒绝'])
        st.info(f"📊 数据核查：当前列表共有 **{pass_count}** 条通过记录，**{fail_count}** 条拒绝记录。")

        # === 4.2 Excel 下载模块 ===
        with st.sidebar:
            st.divider()
            st.header("📥 导出审计详情")
            export_df = main_df.copy()
            
            export_cols = {
                '__ORIGIN__': '关卡ID',
                cm['jid']: '解集ID',
                cm['round_idx']: '测试轮次',   
                cm['diff']: '难度',
                cm['act']: '实际结果',
                cm['hand']: '初始手牌',
                '是否违逆': '是否违逆',
                '违逆类型': '违逆类型'
            }
            
            final_export_cols = {}
            for k, v in export_cols.items():
                if k is not None and k in export_df.columns:
                    final_export_cols[k] = v
                elif v in ['测试轮次']: 
                    if k is None: export_df[v] = 'N/A' 
                    else: final_export_cols[k] = v 

            export_df = export_df.rename(columns=final_export_cols)

            if '测试轮次' not in export_df.columns:
                export_df.insert(2, '测试轮次', range(1, 1 + len(export_df)))
            
            target_cols = ['关卡ID', '解集ID', '测试轮次', '难度', '实际结果', '初始手牌', '是否违逆', '违逆类型']
            target_cols = [c for c in target_cols if c in export_df.columns]
            
            csv_data = export_df[target_cols].to_csv(index=False).encode('utf-8-sig')
            
            st.download_button(
                label="📄 下载逻辑违逆明细 (Excel)",
                data=csv_data,
                file_name="Tripeaks_Logic_Violation_Audit.csv",
                mime="text/csv"
            )
