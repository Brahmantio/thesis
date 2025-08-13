import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
from PIL import Image

st.set_page_config(page_title="House Price Prediction", layout="centered", initial_sidebar_state="auto", page_icon="🏠")
st.title("""
THESIS RESEARCH SIMULATION

OPTIMASI FEATURE ENGINEERING TERHADAP PEFORMA ALGORITMA XGBOOST, RANDOM FOREST DAN SVR DALAM MEMPREDIKSI HARGA RUMAH
\ndashboard was created by [Bramantio](https://www.linkedin.com/in/brahmantio-w/), here I want to try to introduce the results of my portfolio or my abilities in the field of data science. 
\nThis platform aims to provide an introduction, utilization, and exploration resources in the world of machine learning
""")
img = Image.open("rumah1.JPG")
st.image(img, width=500)
    # Membuat tab untuk aplikasi
tab1, tab2 = st.tabs(["Start Prediction", "How to Use"])
with tab1:
        st.header("Input your specific data")
        Cicilan= st.number_input("Cicilan perbulan",
                        min_value=0,
                        step=1000000,
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
        jenis_perumahan= st.radio("Jenis Pemukiman rumah",
                        ["Perumahan", "Perkampungan", "Samping Jalan"])
        kamar_tidur= st.slider("Jumlah kamar tidur",
                        min_value=1,
                        max_value=10,
                        step=1,
                        value=1)
        kamar_mandi= st.slider("Jumlah kamar mandi",
                        min_value=1,
                        max_value=10,
                        step=1,
                        value=1)
        luas_tanah= st.slider("Luas tanah",
                        min_value=16,
                        max_value=5000,
                        step=1,
                        value=2)
        luas_bangunan= st.slider("Luas bangunan",
                        min_value=16,
                        max_value=2000,
                        step=1,
                        value=2)
        Carport= st.slider("Jumlah muat mobil dihalaman",
                        min_value=1,
                        max_value=10,
                        step=1,
                        value=1)
        sertifikat= st.radio("Jenis kepemilikan sertifikat",['SHM', 'SHGB','SHP','SHSRS','PPJB', 'Lainnya'])
        daya_listrik= st.slider("Daya listrik yang tersedia",
                        min_value=100,
                        max_value=66000,
                        step=100,
                        value=100)
        jumlah_lantai= st.slider("jumlah lantai bangunan",
                        min_value=1,
                        max_value=5,
                        step=1,
                        value=1)
        garasi= st.slider("Jumlah muat kendaraaan dalam garasi",
                        min_value=0,
                        max_value=10,
                        step=1,
                        value=0)
        kondisi_properti= st.radio("Tingkat kondisi properti",["Baru","Bagus","Perlu perbaikan","Tidak layak"])
        Dapur = st.slider("Jumlah dapur yang tersedia",
                        min_value=1,
                        max_value=4,
                        step=1,
                        value=1)
        ruang_makan=st.radio("Ketersediaan ruang makan",["Tersedia","Tidak tersedia"])
        ruang_tamu=st.radio("Ketersediaan ruang tamu",["Tersedia","Tidak tersedia"])
        kondisi_perabotan=st.radio("Kondisi fungsional rumah",["Unfurnised","Semi furnished","furnished"])
        material_bangunan=st.radio("material bangunan",["Batako","Bata Hebel","Bata Merah","Beton"])
        material_lantai=st.radio("material lantai",["Granit","Keramik","Marmer","Ubin"])
        hadap=st.radio("Arah rumah",["Barat","Timur","Utara","Selatan"])
        konsep_rumah=st.selectbox('Konsep rumah', ['Modern Glass House', 'Modern', 'Scandinavian', 'Old', 'Mordern minimalist',
                                'Minimalist', 'American Classic', 'Classic','Kontemporer', 'Pavilion','Industrial'])
        pemandangan=st.radio("Pemandangan sekitar",["Pemukiman Warga","Perkotaan","Taman Kota"])
        jangkauan_internet=st.radio("Jangkauan Internet",["Tersedia","Tidak tersedia","Sedang Proses"])
        lebar_jalan=st.slider("Lebar jalan memuat berapa kendaraan",
                        min_value=1,
                        max_value=4,
                        step=1,
                        value=1)
        tahun_bangun=st.date_input('Tahun rumah dibangun')
        tahun_renovasi=st.date_input('Tahun renovasi rumah')
        fasilitas_perumahan=st.multiselect('Fasilitas yang dimiliki', ['Akses parkir','Masjid','Gereja','Taman','Keamanan','One gate system','Kolam renang','Laundry','CCTV'])
        jarak_pusat_kota=st.radio("Berapa jauh jarak dari rumah ke pusat kota",["< 5 KM","5 KM","> 5KM"])
        data = {'Cicilan':Cicilan,
                'Kecamatan':Kecamatan,
                'Wilayah':Wilayah,
                'jenis_perumahan':jenis_perumahan,
                'kamar_tidur':kamar_tidur,
                'kamar_mandi':kamar_mandi,
                'luas_tanah':luas_tanah,
                'luas_bangunan':luas_bangunan,
                'Carport':Carport,
                'sertifikat':sertifikat,
                'daya_listrik':daya_listrik,
                'jumlah_lantai':jumlah_lantai,
                'garasi':garasi,
                'kondisi_properti':kondisi_properti,
                'Dapur':Dapur,
                'ruang_makan':ruang_makan,
                'ruang_tamu':ruang_makan,
                'kondisi_perabotan':kondisi_perabotan,
                'material_bangunan':material_bangunan,
                'material_lantai':material_lantai,
                'hadap':hadap,
                'konsep_rumah':konsep_rumah,
                'pemandangan':pemandangan,
                'jangkauan_internet':jangkauan_internet,
                'lebar_jalan':lebar_jalan,
                'sumber_air':'PDAM',        
                'tahun_bangun':tahun_bangun,
                'tahun_renovasi':tahun_renovasi,
                'fasilitas_perumahan':fasilitas_perumahan,
                'jarak_pusat_kota':jarak_pusat_kota
                }
        features = pd.DataFrame([data])
        # Contoh encoding manual
        Kecamatan={'Lakasantri':0, 
               'Mulyorejo':1, 
               'Kertajaya':2, 
               'Rungkut':3, 
               'Karang Pilang':4,
               'Wiyung':5, 
               'Sukolilo':6,
               'Kenjeran':7, 
               'Tandes':8,
               'Tegalsari':9,
               'Tenggilis Mejoyo':10,
               'Gayungan':11,
               'Dukuh Pakis':12,
               'Sambikerep':13,
               'Jambangan':14,
               'Gunung Anyar':15,
               'Pabean':16,
               'Bulak':17,
               'Wonocolo':18,
               'Wonokromo':19,
               'Sukomanunggal':20,
               'Benowo':21,
               'Semampir':22, 
               'Simokerto':23,
               'Pakal':24,
               'Krembangan':25,
               'Sawahan':26,
               'Tambaksari':27,
               'Genteng':28,
               'Asemrowo':29,
               'Bubutan':30
           }
        Wilayah = {'Surabaya Barat':0, 'Surabaya Timur':1, 'Surabaya Selatan':2, 'Surabaya Utara':3}
        jenis_perumahan = {'Perumahan':0,'Perkampungan':1,'Samping Jalan':2}
        sertifikat = {'SHM':0,'SHGB':1,'SHP':2,'SHSRS':3,'PPJB':4,'Lainnya':5}
        kondisi_properti = {'Baru': 0, 'Bagus': 1, 'Perlu perbaikan': 2, 'Tidak layak':3}
        ruang_makan = {'Tersedia': 0, 'Tidak tersedia': 1}
        ruang_tamu = {'Tersedia': 0, 'Tidak tersedia': 1}
        kondisi_perabotan = {'Unfurnished': 0,'Semi furnished':1, 'furnished': 2}
        material_bangunan = {'Batako':0, 'Bata Hebel': 1, 'Bata Merah': 2,'Beton': 3}
        material_lantai = {'Granit': 0,'Keramik': 1, 'Marmer': 2, 'Ubin':3}
        hadap = {'Barat': 0, 'Timur': 1, 'Utara': 2, 'Selatan':3}
        konsep_rumah = {'Minimalist': 0, 'Kontemporer': 1, 'American Classic': 2,'Modern Glass House':3,'Mordern minimalist':4,'Scandinavian':5,'Pavilion':6,'Industrial':7}
        pemandangan = {'Pemukiman Warga': 0, 'Perkotaan': 1,'Taman Kota':2}
        jangkauan_internet = {'Tersedia': 0, 'Tidak tersedia': 1, 'Sedang proses':2}
        sumber_air = {'PDAM': 0, 'Air sumur':1, 'PAM': 2}
        jarak_pusat_kota = {'< 5 KM': 0, '5 KM': 1, '> 5KM': 2}

        features['Kecamatan'] = features['Kecamatan'].map(Kecamatan)
        features['Wilayah'] = features['Wilayah'].map(Wilayah)
        features['sertifikat'] = features['sertifikat'].map(sertifikat)
        features['kondisi_properti'] = features['kondisi_properti'].map(kondisi_properti)
        features['ruang_makan'] = features['ruang_makan'].map(ruang_makan)
        features['ruang_tamu'] = features['ruang_tamu'].map(ruang_tamu)
        features['kondisi_perabotan'] = features['kondisi_perabotan'].map(kondisi_perabotan)
        features['material_bangunan'] = features['material_bangunan'].map(material_bangunan)
        features['material_lantai'] = features['material_lantai'].map(material_lantai)
        features['hadap'] = features['hadap'].map(hadap)
        features['konsep_rumah'] = features['konsep_rumah'].map(konsep_rumah)
        features['pemandangan'] = features['pemandangan'].map(pemandangan)
        features['jangkauan_internet'] = features['jangkauan_internet'].map(jangkauan_internet)
        features['sumber_air'] = features['sumber_air'].map(sumber_air)
        features['jenis_perumahan'] = features['jenis_perumahan'].map(jenis_perumahan)
        features['jarak_pusat_kota'] = features['jarak_pusat_kota'].map(jarak_pusat_kota)

        #membuat kategori baru dengan satuan tahunan
        features['tahun_bangun'] = pd.to_datetime(features['tahun_bangun'])
        features['tahunbangunan'] = features['tahun_bangun'].dt.year
        features['tahun_renovasi'] = pd.to_datetime(features['tahun_renovasi'])
        features['tahunrenovasi'] = features['tahun_renovasi'].dt.year

        #mengubah entri data yang lebih dari satu menjadi jumlah
        features['jumlah_fasilitas'] = features['fasilitas_perumahan'].apply(lambda x: len(str(x).split(',')))

        # feature combination
        features['efisiensi_ruangan'] = features['luas_bangunan'] / features['luas_tanah']
        features['kualitas_bangunan'] = (features['kondisi_properti'] + features['material_bangunan'] + features['material_lantai'] + features['konsep_rumah']) / 4
        from datetime import datetime
        tahun_sekarang = datetime.now().year
        features['usia_bangunan'] = tahun_sekarang - features['tahunbangunan']
        features['kualitas_infrastruktur'] = (features['sumber_air'] + features['jangkauan_internet'] + features['lebar_jalan'] + features['jarak_pusat_kota']) / 4

            # Predict Button
            if st.button('Predict Now!'):
            #model_loc = '/mount/src/course/modeldqlab.pkl'
             with open("tesis.pkl","rb") as file:
                st.write(features)
                model = pickle.load(file)
                prediction1 = model.predict(features)
                prediction = np.expm1(prediction1)
             with st.spinner('Wait for it...'):
                time.sleep(4)
                st.success(f"Hasil prediksi: Rp{prediction[0]:,.2f}")
     
with tab2:
        st.header("How to use this application")
        st.write("1. Apabila ingin  memprediksi menggunakan file, pastikan file tersebut dalam format .csv dan seluruh atribut sama")
        st.write("2. Supaya prediksi akurat, pastikan nilai yang diinput sudah benar atau sesuai dengan perhitungan")
        st.write("3. Apabila sudah terisi sesuai dengan atribut, tekan tombol 'Predict Now!' untuk memulai")
        st.write("4. Hasil output berupa keterangan nominal harga dalam satuan rupiah")
