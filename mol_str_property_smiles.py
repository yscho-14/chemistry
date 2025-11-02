import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
from rdkit.Chem import Crippen # LogP 계산을 위해 추가
from PIL import Image
import py3Dmol  # 3D 뷰어를 위해 py3Dmol 직접 임포트
import streamlit.components.v1 as components # HTML 렌더링을 위해 임포트
import pandas as pd

# RDKit 오류 로그 비활성화
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

# --- 1. 분자 객체 생성 함수 ---

def get_mol_from_input(input_string, input_type):
    """
    SMILES 또는 InChI 문자열로부터 RDKit Mol 객체를 생성합니다.
    """
    mol = None
    try:
        if input_type == "SMILES":
            mol = Chem.MolFromSmiles(input_string)
        elif input_type == "InChI":
            mol = Chem.MolFromInchi(input_string)
    except Exception as e:
        st.error(f"분자 객체 생성 중 오류 발생: {e}")
        return None
        
    if mol is None:
        st.error(f"오류: 유효하지 않은 {input_type} 문자열입니다. 입력을 확인해 주세요.")
        return None
        
    return mol

# --- 2. 뷰어 함수 (Mol 객체를 직접 받도록 수정) ---

def mol_to_2d_image(mol):
    """
    RDKit Mol 객체를 2D 이미지로 변환합니다.
    """
    return Draw.MolToImage(mol, size=(350, 350))

def generate_3d_mol_block(mol):
    """
    RDKit Mol 객체로부터 3D 구조를 생성하고 MOL 블록(텍스트)으로 반환합니다.
    """
    mol_with_h = Chem.AddHs(mol)
    
    embed_status = AllChem.EmbedMolecule(mol_with_h, AllChem.ETKDG()) 
    
    if embed_status == -1:
        st.warning("3D 구조 생성에 실패했습니다. 더 간단한 2D 구조 기반으로 시도합니다.")
        AllChem.Compute2DCoords(mol_with_h)
    else:
        try:
            AllChem.UFFOptimizeMolecule(mol_with_h)
        except Exception as e:
            st.info(f"3D 구조 최적화 중 사소한 오류 발생: {e} (표시는 계속 진행)")

    mol_block = Chem.MolToMolBlock(mol_with_h)
    return mol_block

# --- 3. [신규] 분자 특성 계산 함수 ---

def calculate_molecular_properties(mol):
    """
    RDKit Mol 객체로부터 주요 분자 특성을 계산합니다.
    """
    properties = {
        "특성 (Property)": [
            "분자량 (Molecular Weight)",
            "LogP (Octanol-water partition coeff.)",
            "TPSA (Topological Polar Surface Area)",
            "수소 결합 공여체 수 (Num H-Bond Donors)",
            "수소 결합 수용체 수 (Num H-Bond Acceptors)",
            "회전 가능 결합 수 (Num Rotatable Bonds)"
        ],
        "값 (Value)": [
            f"{Descriptors.ExactMolWt(mol):.2f}",
            f"{Crippen.MolLogP(mol):.2f}",
            f"{Descriptors.TPSA(mol):.2f} Å²",
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol)
        ],
        "설명 (Description)": [
            "분자의 총 질량 (g/mol)",
            "지용성/친유성 지표 (ADMET 관련)",
            "분자의 극성 표면적 (약물 투과성 예측)",
            "수소 결합을 제공할 수 있는 원자 수",
            "수소 결합을 받을 수 있는 원자 수",
            "분자의 유연성 지표"
        ]
    }
    return pd.DataFrame(properties)

def display_property_metrics(df):
    """
    계산된 특성을 st.metric으로 표시합니다.
    """
    st.subheader("주요 특성 요약 (Key Properties)")
    cols = st.columns(3)
    # df.values[row_index][column_index]
    cols[0].metric("분자량 (MW)", df.values[0][1])
    cols[1].metric("LogP", df.values[1][1])
    cols[2].metric("TPSA", df.values[2][1])

    cols = st.columns(3)
    cols[0].metric("H-Bond 공여체", df.values[3][1])
    cols[1].metric("H-Bond 수용체", df.values[4][1])
    cols[2].metric("회전 가능 결합", df.values[5][1])


# --- Streamlit 앱 인터페이스 ---

st.set_page_config(layout="wide")
st.title("분자 구조 뷰어 및 특성 예측기 🧪🔬")

# --- 1. [수정] 입력 섹션 ---
st.header("1. 분자 정보 입력")

# 입력 타입 선택 (SMILES 또는 InChI)
input_type = st.radio("입력 타입 선택", ["SMILES", "InChI"])

default_smiles = "CC(=O)Oc1ccccc1C(=O)O" # 아스피린
default_inchi = "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)"

# 선택된 타입에 따라 다른 입력창 표시
if input_type == "SMILES":
    smiles_input = st.text_input(
        "SMILES 문자열을 입력하세요:", 
        default_smiles
    )
    user_input = smiles_input
elif input_type == "InChI":
    inchi_input = st.text_input(
        "InChI 문자열을 입력하세요:", 
        default_inchi
    )
    user_input = inchi_input

mol = None
if user_input:
    # 유효성 검사를 포함하여 Mol 객체 생성
    mol = get_mol_from_input(user_input, input_type)

# Mol 객체가 성공적으로 생성되었을 때만 하위 섹션 표시
if mol:
    # --- 2. [신규] 특성 예측 섹션 ---
    st.header("2. 분자 특성 예측")
    try:
        properties_df = calculate_molecular_properties(mol)
        
        # st.metric으로 요약 대시보드 표시
        display_property_metrics(properties_df)
        
        # st.dataframe으로 상세 테이블 표시
        st.subheader("특성 상세 (All Properties)")
        st.dataframe(properties_df, use_container_width=True, hide_index=True)
        
        st.info(
            "본 예측기는 RDKit의 Descriptor를 사용하여 계산된 값입니다. "
            "용해도(Solubility), 끓는점(BP), 녹는점(MP) 등은 별도의 학습된 ML 모델이 필요합니다."
        )
        
    except Exception as e:
        st.error(f"특성 계산 중 오류가 발생했습니다: {e}")

    # --- 3. [수정] 뷰어 섹션 ---
    st.header("3. 분자 구조 뷰어")
    
    # 2D와 3D 뷰를 위한 탭 생성
    tab1, tab2 = st.tabs(["2D 분자 구조", "3D 인터랙티브 구조"])

    # --- 2D 구조 탭 ---
    with tab1:
        st.subheader("2D 분자 구조")
        try:
            img = mol_to_2d_image(mol)
            st.image(img, caption="생성된 2D 분자 구조")
        except Exception as e:
            st.error(f"2D 이미지 생성 중 오류가 발생했습니다: {e}")

    # --- 3D 구조 탭 ---
    with tab2:
        st.subheader("3D 인터랙티브 분자 구조")
        st.markdown("마우스 휠로 **줌(Zoom)**, 드래그로 **회전(Rotate)**이 가능합니다.")
        
        # [수정됨] "line"과 "cross" 옵션 제거
        style_3d = st.selectbox("3D 표시 스타일 선택", ["stick", "sphere"])
        
        try:
            mol_block_3d = generate_3d_mol_block(mol)
            
            # 1. py3Dmol 뷰어 객체 생성
            view = py3Dmol.view(width=650, height=450)
            view.addModel(mol_block_3d, 'mol')
            view.setStyle({style_3d: {}})
            view.zoomTo()
            
            # 2. 뷰어를 HTML로 변환
            html_3d = view._make_html()
            
            # 3. Streamlit HTML 컴포넌트로 렌더링
            components.html(html_3d, width=650, height=450)
            
        except Exception as e:
            st.error(f"3D 뷰어 생성 중 오류가 발생했습니다: {e}")

# 간단한 사용법 안내
st.sidebar.title("사용법")
st.sidebar.markdown("""
1.  **입력 타입 선택**: 'SMILES' 또는 'InChI' 중 하나를 선택합니다.
2.  **문자열 입력**: 선택한 타입의 문자열을 텍스트 상자에 입력하거나 붙여넣습니다. (예: 아스피린)
3.  **특성 확인**: '2. 분자 특성 예측' 섹션에서 계산된 분자량, LogP, TPSA 등의 주요 물성 정보를 확인합니다.
4.  **구조 확인**: '3. 분자 구조 뷰어' 섹션에서 '2D' 또는 '3D' 탭을 선택하여 분자 구조를 시각적으로 확인합니다.
""")
