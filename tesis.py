import streamlit as st
import pandas as pd
import pickle
import time
from PIL import Image

st.set_page_config(page_title="House Price Prediction", layout="centered", initial_sidebar_state="auto", page_icon="🏠")
st.title("""
Welcome to my portofolio Data Analyst

OPTIMASI FEATURE ENGINEERING TERHADAP PEFORMA ALGORITMA XGBOOST, RANDOM FOREST DAN SVR DALAM MEMPREDIKSI HARGA RUMAH
\ndashboard was created by [Bramantio](https://www.linkedin.com/in/brahmantio-w/), here I want to try to introduce the results of my portfolio or my abilities in the field of data science. 
\nThis platform aims to provide an introduction, utilization, and exploration resources in the world of machine learning
""")
img = Image.open("rumah1.JPG")
st.image(img, width=500)
add_selectitem = st.sidebar.header("Prediction with CSV file")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
        
    # Membuat tab untuk aplikasi
tab1, tab2 = st.tabs(["Start Prediction", "How to Use"])
with tab1:
        st.header("Input your specific data")
        cicilan= st.number_input("Cicilan perbulan",
                        min_value=0,
                        step=1000000000,
                        )
        Kecamatan = st.selectbox('Kecamatan rumah anda:', 
                                   ['Lakasantri', 'Mulyorejo', 'Kertajaya', 'Rungkut', 'Karang Pilang',
                                    'Wiyung', 'Sukolilo', 'Kenjeran', 'Tandes', 'Tegalsari',
                                    'Tenggilis Mejoyo', 'Gayungan', 'Dukuh Pakis', 'Sambikerep',
                                    'Jambangan', 'Gunung Anyar', 'Pabean', 'Bulak', 'Wonocolo',
                                    'Wonokromo', 'Sukomanunggal', 'Benowo', 'Semampir', 'Simokerto',
                                    'Pakal', 'Krembangan', 'Sawahan', 'Tambaksari', 'Genteng',
                                    'Asemrowo', 'Bubutan'])
        Wilayah= st.selectbox('Pilih Wilayah:', ['Surabaya Barat', 'Surabaya Timur', 'Surabaya Selatan',
                            'Surabaya Utara', 'Surabaya Pusat'])
        Pemukiman= st.radio("Jenis Pemukiman rumah",
                        ("Perumahan", "Perkampungan", "Samping Jalan"))
        kamartidur= st.slider("Jumlah kamar tidur",
                        min_value=1,
                        max_value=10,
                        step=1,
                        value=1)
        kamarmandi= st.slider("Jumlah kamar mandi",
                        min_value=1,
                        max_value=10,
                        step=1,
                        value=1)
        Luastanah= st.slider("Luas tanah",
                        min_value=16,
                        max_value=5000,
                        step=1,
                        value=2)
        Luasbangunan= st.slider("Luas bangunan",
                        min_value=16,
                        max_value=2000,
                        step=1,
                        value=2)
        carport= st.slider("Jumlah muat mobil dihalaman",
                        min_value=1,
                        max_value=10,
                        step=1,
                        value=1)
        sertifikat= st.radio("Jenis kepemilikan sertifikat",('SHM', 'SHGB', 'PPJB', 'SHP','Lainnya'))
        dayalistrik= st.slider("Daya listrik yang tersedia",
                        min_value=100,
                        max_value=66000,
                        step=100,
                        value=100)
        garasi= st.slider("Jumlah muat kendaraaan dalam garasi",
                        min_value=0,
                        max_value=10,
                        step=1,
                        value=0)
        kondisi= st.radio("Tingkat kondisi properti",("Baru","Bagus","Perlu perbaikan","Tidak layak"))
        dapur = st.slider("Jumlah dapur yang tersedia",
                        min_value=1,
                        max_value=4,
                        step=1,
                        value=1)
        Ruangmakan=st.radio("Ketersediaan ruang makan",("Tersedia","Tidak tersedia"))
        Ruangtamu=st.radio("Ketersediaan ruang tamu",("Tersedia","Tidak tersedia"))
        perabotan=st.radio("Kondisi fungsional rumah",("Unfurnised","Semi furnished","furnished"))
        materialbangunan=st.radio("material bangunan",("Batako","Bata Hebel","Bata Merah","Beton"))
        materialantai=st.radio("material lantai",("Granit","Keramik","Marmer","Ubin"))
        hadap=st.radio("Arah rumah",("Barat","Timur","Utara","Selatan"))
        konseprumah=st.selectbox('Konsep rumah', ['Modern Glass House', 'Modern', 'Scandinavian', 'Old', 'Mordern minimalist',
                                'Minimalist', 'American Classic', 'Classic','Kontemporer', 'Pavilion','Industrial'])
        pemandangan=st.radio("Pemandangan sekitar",("Pemukiman Warga","Perkotaan","Taman Kota"))
        internet=st.radio("Jangkauan Internet",("Tersedia","Tidak tersedia","Sedang Proses"))
        jalan=st.slider("Lebar jalan memuat berapa kendaraan",
                        min_value=1,
                        max_value=4,
                        step=1,
                        value=1)
        tahunbangun=st.date_input('Tahun rumah dibangun')
        tahunrenov=st.date_input('Tahun renovasi rumah')
        fasilitas=st.multiselect('Fasilitas yang dimiliki', ['Akses parkir','Masjid','Gereja','Taman','Keamanan','One gate system','Kolam renang','Laundry','CCTV'])
        jarakkota=st.radio("Berapa jauh jarak dari rumah ke pusat kota",("< 5 KM","5 KM","> 5KM"))
        data = {'cicilan':cicilan,
                'Kecamatan':Kecamatan,
                'Wilayah':Wilayah,
                'Pemukiman':Pemukiman,
                'kamartidur':kamartidur,
                'kamarmandi':kamarmandi,
                'Luastanah':Luastanah,
                'Luasbangunan':Luasbangunan,
                'carport':carport,
                'sertifikat':sertifikat,
                'dayalistrik':dayalistrik,
                'garasi':garasi,
                'kondisi':kondisi,
                'dapur':dapur,
                'Ruangmakan':Ruangmakan,
                'Ruangtamu':Ruangtamu,
                'perabotan':perabotan,
                'materialbangunan':materialbangunan,
                'materialantai':materialantai,
                'hadap':hadap,
                'konseprumah':konseprumah,
                'pemandangan':pemandangan,
                'internet':internet,
                'jalan':jalan,
                'tahunbangun':tahunbangun.year,
                'tahunrenov':tahunrenov.year,
                'fasilitas':','.join(fasilitas) if fasilitas else 'Tidak ada',
                'jarakkota':jarakkota
                }
        features = pd.DataFrame(data, index=[0])
        features = features.drop(columns=['Luastanah','Luasbangunan','Konseprumah','tahunbangun','fasilitas'], errors='ignore')

    # Predict Button
if st.button('Predict Now!'):
            #model_loc = '/mount/src/course/modeldqlab.pkl'
 with open("tesis.pkl",'rb') as file:
        model = pickle.load(file)
        prediction1 = model.predict(features)
        prediction = np.expm1(prediction1)
 with st.spinner('Wait for it...'):
        time.sleep(4)
        st.success(f"Hasil prediksi: ${prediction[0]:,.2f}")
        
with tab2:
        st.header("How to use this application")
        st.write("1. Apabila ingin  memprediksi menggunakan file, pastikan file tersebut dalam format .csv dan seluruh atribut sama")
        st.write("2. Supaya prediksi akurat, pastikan nilai yang diinput sudah benar atau sesuai dengan perhitungan")
        st.write("3. Apabila sudah terisi sesuai dengan atribut, tekan tombol 'Predict Now!' untuk memulai")
        st.write("4. Hasil output berupa keterangan nominal harga dalam satuan rupiah")
