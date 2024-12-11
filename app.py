import streamlit as st
import pandas as pd
import numpy as np
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import ks_2samp
import time
from io import StringIO
import base64

def get_csv_download_link(df, filename="data.csv"):
    """DataFrameをダウンロードリンクとして提供"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def generate_test_data():
    """テストデータの生成"""
    np.random.seed(42)
    n = 1000
    
    # 年齢と経験年数に相関を持たせる
    age = np.random.normal(35, 10, n)
    experience = np.maximum(0, age - 22 + np.random.normal(0, 2, n))
    
    # 収入を年齢と経験年数から計算
    base_income = 30000 + experience * 2000 + (age - 25) * 500
    income = np.maximum(25000, base_income + np.random.normal(0, 5000, n))
    
    df = pd.DataFrame({
        'age': np.round(age, 1),
        'years_experience': np.round(experience, 1),
        'annual_income': np.round(income, -2)  # 100単位で丸める
    })
    
    # 現実的な範囲に収める
    df['age'] = df['age'].clip(22, 65)
    df['years_experience'] = df['years_experience'].clip(0, 40)
    df['annual_income'] = df['annual_income'].clip(25000, 150000)
    
    return df

def validate_data(df):
    """データの検証と警告の表示"""
    warnings = []
    errors = []
    
    # サイズチェック
    if len(df) > 100000:
        errors.append("データサイズが大きすぎます（最大100,000行まで）")
    
    # 欠損値チェック
    if df.isnull().any().any():
        warnings.append("欠損値が含まれています。自動的に処理されます。")
    
    # 数値型列の確認
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        errors.append("少なくとも2つの数値型列が必要です")
    
    return warnings, errors

def load_data(uploaded_file):
    """アップロードされたCSVファイルを読み込む"""
    if uploaded_file is not None:
        try:
            string_data = StringIO(uploaded_file.getvalue().decode('utf-8'))
            df = pd.read_csv(string_data)
            warnings, errors = validate_data(df)
            
            for warning in warnings:
                st.warning(warning)
            for error in errors:
                st.error(error)
                return None
                
            return df
        except Exception as e:
            st.error(f"データ読み込みエラー: {str(e)}")
            return None
    return None

@st.cache_data
def generate_synthetic_data(real_data, num_rows, method='gaussian_copula', epochs=100):
    """選択された手法で合成データを生成する（キャッシュ付き）"""
    try:
        # メタデータの作成
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(real_data)
        
        if method == 'gaussian_copula':
            model = GaussianCopulaSynthesizer(metadata)
            st.info('GaussianCopulaを使用してデータを生成中...')
        else:  # CTGAN
            model = CTGANSynthesizer(metadata, epochs=epochs)
            st.info(f'CTGANを使用してデータを生成中... (エポック数: {epochs})')
        
        start_time = time.time()
        
        # モデルの学習とサンプリング
        model.fit(real_data)
        synthetic_data = model.sample(num_rows)
        generation_time = time.time() - start_time
        
        return synthetic_data, generation_time
        
    except Exception as e:
        st.error(f"データ生成エラー: {str(e)}")
        return None, 0

def calculate_distribution_similarity(real_data, synthetic_data, column):
    """KSテストを使用して分布の類似度を計算"""
    statistic, pvalue = ks_2samp(real_data[column], synthetic_data[column])
    return 1 - statistic  # 類似度に変換（1に近いほど類似）

def plot_distribution_comparison(real_data, synthetic_data, column):
    """分布比較のプロット生成"""
    fig = go.Figure()
    
    # 実データのヒストグラム
    fig.add_trace(go.Histogram(
        x=real_data[column],
        name='Real Data',
        opacity=0.7,
        nbinsx=30
    ))
    
    # 合成データのヒストグラム
    fig.add_trace(go.Histogram(
        x=synthetic_data[column],
        name='Synthetic Data',
        opacity=0.7,
        nbinsx=30
    ))
    
    fig.update_layout(
        barmode='overlay',
        title=f'Distribution Comparison: {column}',
        xaxis_title=column,
        yaxis_title='Count'
    )
    
    return fig

def calculate_correlation_matrices(real_data, synthetic_data):
    """相関行列の計算と比較"""
    real_corr = real_data.corr()
    synthetic_corr = synthetic_data.corr()
    correlation_diff = abs(real_corr - synthetic_corr)
    return real_corr, synthetic_corr, correlation_diff

def plot_correlation_matrices(real_corr, synthetic_corr, correlation_diff):
    """相関行列の比較プロット"""
    fig = go.Figure()
    
    # ボタンで切り替え可能な3つのヒートマップ
    figures = [
        ('Real Data', real_corr),
        ('Synthetic Data', synthetic_corr),
        ('Absolute Difference', correlation_diff)
    ]
    
    for i, (name, matrix) in enumerate(figures):
        fig.add_trace(go.Heatmap(
            z=matrix,
            x=matrix.columns,
            y=matrix.columns,
            colorscale='RdBu' if i < 2 else 'Reds',
            zmin=-1 if i < 2 else 0,
            zmax=1 if i < 2 else 1,
            name=name,
            visible=i == 0  # 最初は実データのみ表示
        ))
    
    # ボタンの設定
    buttons = []
    for i, (name, _) in enumerate(figures):
        visible = [j == i for j in range(len(figures))]
        buttons.append(dict(
            label=name,
            method="update",
            args=[{"visible": visible}]
        ))
    
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.7,
            y=1.15,
            showactive=True,
            buttons=buttons
        )],
        title="Correlation Matrix Comparison"
    )
    
    return fig

def display_results():
    """評価結果の表示（セッションステートからデータを使用）"""
    if 'synthetic_data' not in st.session_state:
        return

    synthetic_data = st.session_state['synthetic_data']
    real_data = st.session_state['real_data']
    generation_time = st.session_state['generation_time']

    st.success(f'合成データの生成が完了しました！ (処理時間: {generation_time:.2f}秒)')
    
    # 合成データのダウンロードリンク
    st.markdown(
        get_csv_download_link(synthetic_data, "synthetic_data.csv"),
        unsafe_allow_html=True
    )
    
    # 評価結果の表示
    st.header('3. 評価結果')
    
    # 分布の比較
    st.subheader('3.1 分布の比較')
    numeric_columns = real_data.select_dtypes(include=[np.number]).columns
    selected_column = st.selectbox('列を選択', numeric_columns)
    
    similarity = calculate_distribution_similarity(
        real_data, synthetic_data, selected_column
    )
    st.metric(
        "分布の類似度 (1に近いほど類似)",
        f"{similarity:.3f}"
    )
    
    fig = plot_distribution_comparison(
        real_data, synthetic_data, selected_column
    )
    st.plotly_chart(fig)
    
    # 相関行列の比較
    st.subheader('3.2 相関関係の比較')
    real_corr, synthetic_corr, correlation_diff = calculate_correlation_matrices(
        real_data, synthetic_data
    )
    
    mean_correlation_diff = correlation_diff.mean().mean()
    st.metric(
        "相関の平均差異 (0に近いほど類似)",
        f"{mean_correlation_diff:.3f}"
    )
    
    fig = plot_correlation_matrices(
        real_corr, synthetic_corr, correlation_diff
    )
    st.plotly_chart(fig)

def main():
    st.title('合成データ生成・評価デモ')
    
    # サイドバーにテストデータの説明を配置
    st.sidebar.header("テストデータについて")
    st.sidebar.write("""
    このデモアプリには、テスト用のサンプルデータが含まれています。
    - 1000行のデータ
    - 年齢、経験年数、年収の3つの列
    - 現実的な相関関係を含む
    """)
    
    # テストデータの生成とダウンロードリンク
    test_data = generate_test_data()
    st.sidebar.markdown(get_csv_download_link(test_data, "test_data.csv"), unsafe_allow_html=True)
    
    # コンテナを使用して各セクションを分離
    data_container = st.container()
    settings_container = st.container()
    results_container = st.container()
    
    with data_container:
        st.header('1. データの準備')
        data_input = st.radio(
            "データの入力方法を選択してください：",
            ['テストデータを使用', 'CSVファイルをアップロード']
        )
        
        real_data = None
        if data_input == 'テストデータを使用':
            real_data = test_data
            st.success("テストデータを読み込みました")
        else:
            uploaded_file = st.file_uploader("CSVファイルをアップロード", type=['csv'])
            if uploaded_file is not None:
                real_data = load_data(uploaded_file)
        
        if real_data is not None:
            st.write("データの形状:", real_data.shape)
            st.write("サンプルデータ:", real_data.head())
    
    if real_data is not None:
        with settings_container:
            # 合成手法の選択
            st.header('2. 合成手法とパラメータの設定')
            col1, col2 = st.columns(2)
            
            with col1:
                method = st.selectbox(
                    '合成手法を選択',
                    ['gaussian_copula', 'ctgan'],
                    format_func=lambda x: {
                        'gaussian_copula': 'GaussianCopula (高速・シンプル)',
                        'ctgan': 'CTGAN (高品質・低速)'
                    }[x]
                )
            
            with col2:
                num_rows = st.number_input(
                    '生成する行数',
                    min_value=1,
                    max_value=min(100000, len(real_data) * 2),
                    value=len(real_data)
                )
            
            # CTGANのパラメータ設定
            epochs = 50  # デフォルト値
            if method == 'ctgan':
                # エポック数を直接指定できるスライダーに変更
                epochs = st.slider(
                    'エポック数',
                    min_value=10,
                    max_value=10000,
                    value=100,
                    format="%d",  # 整数値として表示
                )
                st.info(f"""
                エポック数の目安：
                - 100以下：高速だが品質は低め
                - 100-1000：バランスの取れた設定
                - 1000以上：高品質だが処理時間が長い
                
                大きなデータセットでは少なめに設定することをお勧めします。
                処理時間の目安：
                - 1000行のデータで100エポック ≒ 2-3分
                - 1000行のデータで1000エポック ≒ 20-30分
                """)
            
            if st.button('合成データを生成', help='クリックするとデータ生成が始まります'):
                with st.spinner('データを生成中...'):
                    synthetic_data, generation_time = generate_synthetic_data(
                        real_data, num_rows, method, epochs
                    )
                    
                    if synthetic_data is not None:
                        # セッションステートにデータを保存
                        st.session_state['synthetic_data'] = synthetic_data
                        st.session_state['real_data'] = real_data
                        st.session_state['generation_time'] = generation_time
    
    # 評価結果の表示（別のコンテナ内）
    with results_container:
        if 'synthetic_data' in st.session_state:
            display_results()

if __name__ == '__main__':
    main()
