import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from PIL import Image
import py3Dmol  # 3D 뷰어를 위해 py3Dmol 직접 임포트
import streamlit.components.v1 as components # HTML 렌더링을 위해 임포트

# RDKit 오류 로그 비활성화
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

def smiles_to_2d_image(smiles_string):
    """
    SMILES 문자열을 2D 이미지로 변환합니다.
    """
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=(350, 350))

def generate_3d_mol_block(smiles_string):
    """
    SMILES 문자열로부터 3D 구조를 생성하고 MOL 블록(텍스트)으로 반환합니다.
    """
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None
    
    # 3D 구조 생성을 위해 수소 원자 추가
    mol_with_h = Chem.AddHs(mol)
    
    # 3D 구조 생성 (ETKDG 알고리즘 사용)
    embed_status = AllChem.EmbedMolecule(mol_with_h, AllChem.ETKDG()) 
    
    if embed_status == -1:
        st.warning("3D 구조 생성에 실패했습니다. 더 간단한 2D 구조 기반으로 시도합니다.")
        AllChem.Compute2DCoords(mol_with_h)
    else:
        # 3D 구조 생성 성공 시, 구조 최적화
        try:
            AllChem.UFFOptimizeMolecule(mol_with_h)
        except Exception as e:
            st.info(f"3D 구조 최적화 중 사소한 오류 발생: {e} (표시는 계속 진행)")

    # 3D 뷰어(stmol)가 읽을 수 있도록 MOL 블록 형식으로 변환
    mol_block = Chem.MolToMolBlock(mol_with_h)
    return mol_block

# --- Streamlit 앱 인터페이스 ---

st.set_page_config(layout="wide") # 페이지 레이OUT을 넓게 사용
st.title("SMILES 분자 구조 뷰어 (2D & 3D) 🧪🔬")

# 사용자로부터 SMILES 문자열 입력받기
st.subheader("SMILES 문자열을 입력하세요:")
smiles_input = st.text_input(
    "예: CC(=O)Oc1ccccc1C(=O)O (아스피린)", 
    "CC(=O)Oc1ccccc1C(=O)O"
)

if smiles_input:
    # 1. SMILES 유효성 검사
    mol_check = Chem.MolFromSmiles(smiles_input)
    
    if mol_check is None:
        st.error("오류: 유효하지 않은 SMILES 문자열입니다. 입력을 확인해 주세요.")
    else:
        # 2. 2D와 3D 뷰를 위한 탭 생성
        tab1, tab2 = st.tabs(["2D 분자 구조", "3D 인터랙티브 구조"])

        # --- 2D 구조 탭 ---
        with tab1:
            st.subheader("2D 분자 구조")
            try:
                img = smiles_to_2d_image(smiles_input)
                if img:
                    st.image(img, caption="생성된 2D 분자 구조")
                else:
                    st.error("2D 이미지 생성에 실패했습니다.")
            except Exception as e:
                st.error(f"2D 이미지 생성 중 오류가 발생했습니다: {e}")

        # --- 3D 구조 탭 ---
        with tab2:
            st.subheader("3D 인터랙티브 분자 구조")
            st.markdown("마우스 휠로 **줌(Zoom)**, 드래그로 **회전(Rotate)**이 가능합니다.")
            
            # 3D 스타일 선택 기능
            style_3d = st.selectbox("3D 표시 스타일 선택", ["stick", "line", "cross", "sphere"])
            
            try:
                # 3D MOL 블록 생성
                mol_block_3d = generate_3d_mol_block(smiles_input)
                
                if mol_block_3d:
                    
                    # 1. py3Dmol 뷰어 객체 생성
                    view = py3Dmol.view(width=650, height=450)
                    
                    # 2. 뷰어에 분자 데이터(MOL 블록) 추가
                    view.addModel(mol_block_3d, 'mol')
                    
                    # 3. 선택된 스타일 적용
                    view.setStyle({style_3d: {}})
                    
                    # 4. 분자가 뷰어에 꽉 차도록 줌 설정
                    view.zoomTo()
                    
                    # 5. py3Dmol 뷰어를 HTML로 변환
                    html_3d = view._make_html()
                    
                    # 6. st.components.v1.html을 사용하여 HTML 렌더링
                    components.html(html_3d, width=650, height=450)
                    
                else:
                    st.error("3D 분자 데이터 생성에 실패했습니다.")
                    
            except Exception as e:
                st.error(f"3D 뷰어 생성 중 오류가 발생했습니다: {e}")

# 간단한 사용법 안내
st.markdown("""
---
### 사용법
1.  위의 텍스트 상자에 분자의 [SMILES](https://ko.wikipedia.org/wiki/SMILES) 표기법을 입력합니다.
2.  **'2D 분자 구조'** 탭에서 2차원 이미지를 확인합니다.
3.  **'3D 인터랙티브 구조'** 탭에서 3차원 분자 모델을 확인하고 마우스로 조작합니다.
""")
