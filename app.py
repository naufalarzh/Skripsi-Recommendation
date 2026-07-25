import streamlit as st
import pandas as pd
import numpy as np
import joblib
from fpdf import FPDF

# Set konfigurasi halaman utama website
st.set_page_config(
    page_title="Sistem Rekomendasi Latihan Beban Hibrida",
    page_icon="🏋️‍♂️",
    layout="wide",
)


# TAHAP 1 & 2: MEMUAT DATASET BERSIH DAN MODEL PKL (Menggunakan Cache Streamlit)
@st.cache_resource
def load_resources():
    # Memuat model terbaik hasil latihan
    model = joblib.load("model_klasifikasi_bodypart.pkl")
    # Memuat dataset yang sudah dibersihkan final
    df = pd.read_csv("exercise_clean_final.csv")
    return model, df


try:
    model, df = load_resources()
except Exception as e:
    st.error(
        f"❌ Gagal memuat file! Pastikan 'model_klasifikasi_bodypart.pkl' dan 'exercise_clean_final.csv' berada di folder yang sama dengan app.py. Error: {e}"
    )
    st.stop()

# ==========================================================
# ATURAN BISNIS / CONSTRAINT RULES (Business Understanding)
# ==========================================================
goal_map = {
    "Muscle Gain": {
        "preferred_types": ["Strength", "Powerlifting", "Strongman"],
        "sets": "3-4 set",
        "reps": "8-12 repetisi",
    },
    "Fat Loss": {
        "preferred_types": ["Strength", "Cardio", "Plyometrics"],
        "sets": "3 set",
        "reps": "12-15 repetisi",
    },
    "Strength": {
        "preferred_types": ["Strength", "Powerlifting", "Olympic Weightlifting"],
        "sets": "4-5 set",
        "reps": "4-8 repetisi",
    },
}

# Template pembagian latihan (Split) berdasarkan jumlah hari per minggu
split_templates = {
    2: ["Upper Body", "Lower Body"],
    3: ["Push", "Pull", "Legs"],
    4: ["Chest/Shoulders/Triceps", "Back/Biceps", "Legs/Core", "Full Body"],
    5: ["Chest", "Back", "Legs", "Shoulders/Arms", "Core/Conditioning"],
}

# Batasan tingkat kemahiran (Level) yang diizinkan untuk keamanan user
allowed_level_map = {
    "Beginner": ["Beginner"],
    "Intermediate": ["Beginner", "Intermediate"],
    "Expert": ["Intermediate", "Expert"],
}

# Prioritas target otot (BodyPart) untuk setiap fokus harian
bodypart_priority_map = {
    "Upper Body": ["Chest", "Shoulders", "Triceps", "Biceps", "Lats", "Middle Back"],
    "Lower Body": ["Quadriceps", "Hamstrings", "Glutes", "Calves"],
    "Push": ["Chest", "Shoulders", "Triceps"],
    "Pull": ["Lats", "Middle Back", "Biceps", "Lower Back", "Traps"],
    "Legs": ["Quadriceps", "Hamstrings", "Glutes", "Calves"],
    "Chest/Shoulders/Triceps": ["Chest", "Shoulders", "Triceps"],
    "Back/Biceps": ["Lats", "Middle Back", "Biceps", "Lower Back", "Traps"],
    "Legs/Core": ["Quadriceps", "Hamstrings", "Glutes", "Calves", "Abdominals"],
    "Full Body": [
        "Chest",
        "Lats",
        "Shoulders",
        "Quadriceps",
        "Hamstrings",
        "Glutes",
        "Abdominals",
    ],
    "Chest": ["Chest"],
    "Back": ["Lats", "Middle Back", "Lower Back", "Traps"],
    "Shoulders/Arms": ["Shoulders", "Biceps", "Triceps", "Forearms"],
    "Core/Conditioning": ["Abdominals"],
}

# Aturan Bisnis Baru: Pemetaan tempat latihan ke alat yang tersedia
location_equipment_map = {
    "gym": ["All"],
    "rumah_ada_alat": ["Dumbbell", "Bands", "Body Only", "Exercise Ball", "Mat"],
    "rumah_tanpa_alat": ["Body Only"],
}

available_muscles = sorted(df["BodyPart"].unique().tolist())


# ==========================================================
# FUNGSI MESIN GENERATOR PDF (Format Cetak Eksklusif)
# ==========================================================
def generate_pdf(meta_profile, compiled_schedule):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(31, 41, 55)
    pdf.cell(0, 10, "REKOMENDASI PROGRAM LATIHAN BEBAN HIBRIDA", ln=True, align="C")

    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(107, 114, 128)
    pdf.cell(
        0,
        5,
        "Hasil Pemetaan Kombinasi Aturan Bisnis & Machine Learning Classifier",
        ln=True,
        align="C",
    )
    pdf.ln(4)

    pdf.set_draw_color(29, 78, 216)
    pdf.set_line_width(0.8)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    pdf.set_fill_color(243, 244, 246)
    pdf.rect(10, pdf.get_y(), 190, 24, "F")

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(31, 41, 55)
    pdf.cell(0, 6, "  PARAMETER PROFIL PENGGUNA", ln=True)

    pdf.set_font("Helvetica", "", 9)
    pdf.cell(63, 5, f"  - Tingkat Kemahiran : {meta_profile['level']}", ln=False)
    pdf.cell(63, 5, f"- Target Latihan : {meta_profile['goal']}", ln=False)
    pdf.cell(63, 5, f"- Lokasi Latihan : {meta_profile['location']}", ln=True)
    pdf.cell(63, 5, f"  - Batasan Alat      : {meta_profile['equipment']}", ln=False)
    pdf.cell(63, 5, f"- Volume Set/Rep : {meta_profile['volume']}", ln=True)
    pdf.ln(8)

    for element in compiled_schedule:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(29, 78, 216)
        pdf.cell(
            0,
            6,
            f"{element['day'].upper()} - FOKUS: {element['focus'].upper()}",
            ln=True,
        )
        pdf.ln(1)

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(255, 255, 255)
        pdf.set_fill_color(31, 41, 55)
        pdf.cell(
            75, 7, " Nama Gerakan (Title)", border=1, ln=False, align="L", fill=True
        )
        pdf.cell(30, 7, " Target Otot", border=1, ln=False, align="C", fill=True)
        pdf.cell(32, 7, " Alat (Equipment)", border=1, ln=False, align="C", fill=True)
        pdf.cell(28, 7, " Tipe Latihan", border=1, ln=False, align="C", fill=True)
        pdf.cell(25, 7, " Skor Hibrida", border=1, ln=True, align="C", fill=True)

        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(55, 65, 81)

        idx_zebra = 0
        for _, baris in element["dataframe"].iterrows():
            warna_bg = (249, 250, 251) if idx_zebra % 2 == 0 else (255, 255, 255)
            pdf.set_fill_color(*warna_bg)

            nama_gerakan = str(baris["Title"])
            if len(nama_gerakan) > 42:
                nama_gerakan = nama_gerakan[:39] + "..."

            pdf.cell(
                75, 6, f" {nama_gerakan}", border=1, ln=False, align="L", fill=True
            )
            pdf.cell(
                30, 6, str(baris["BodyPart"]), border=1, ln=False, align="C", fill=True
            )
            pdf.cell(
                32, 6, str(baris["Equipment"]), border=1, ln=False, align="C", fill=True
            )
            pdf.cell(
                28, 6, str(baris["Type"]), border=1, ln=False, align="C", fill=True
            )
            pdf.cell(
                25,
                6,
                f"{baris['final_score']:.4f}",
                border=1,
                ln=True,
                align="C",
                fill=True,
            )
            idx_zebra += 1

        pdf.ln(5)

    return pdf.output()


# ==========================================================
# STRUKTUR ANTARMUKA WEB (STREAMLIT UI)
# ==========================================================
st.title("Sistem Rekomendasi Jadwal & Gerakan Latihan Beban")
st.markdown(
    "Aplikasi web rekomendasi hibrida berbasis **Constraint Rules** dan **Machine Learning** untuk tugas skripsi."
)
st.divider()

col_sidebar, col_content = st.columns([1, 2])

# --- KOLOM KIRI: INPUT PARAMETER USER ---
with col_sidebar:
    st.header("Profil & Batasan Pengguna")

    user_level = st.selectbox(
        "Tingkat Kemahiran Anda (Level):", ["Beginner", "Intermediate", "Expert"]
    )
    user_goal = st.selectbox("Tujuan Latihan (Goal):", list(goal_map.keys()))
    user_location = st.radio("Tempat Latihan:", ["Gym", "Rumah"])

    # Logika untuk menampilkan pilihan alat jika memilih Rumah
    has_equipment = True
    if user_location == "Rumah":
        has_equipment_ui = st.radio(
            "Ketersediaan Alat di Rumah:",
            ["Ada Alat Latihan", "Tidak Ada Alat"],
        )
        has_equipment = (
            True if has_equipment_ui == "Ada Alat Latihan (Dumbbell/Bands)" else False
        )
    else:
        st.caption(
            "Skenario Gym otomatis membuka akses ke **Semua Alat** yang ada di dataset."
        )

    st.caption(
        "Ketersediaan alat akan disesuaikan secara otomatis berdasarkan pilihan lokasi latihan Anda."
    )

    st.markdown("---")
    st.subheader("Pengaturan Pembagian Otot")
    days_per_week = st.number_input(
        "Jumlah Hari Latihan per Minggu:", min_value=2, max_value=5, value=3
    )
    exercises_per_day = st.slider(
        "Target Jumlah Gerakan per Hari:", min_value=1, max_value=10, value=6
    )

    custom_schedule = {}
    st.caption(
        "Pilihan otot otomatis diisi berdasarkan standar split (bisa diubah manual):"
    )
    current_split = split_templates.get(
        days_per_week, [f"Fokus" for _ in range(days_per_week)]
    )

    for d, split_name in enumerate(current_split, start=1):
        recommended_muscles = bodypart_priority_map.get(split_name, [])
        default_muscles = [m for m in recommended_muscles if m in available_muscles]

        selected_muscles = st.multiselect(
            f"Hari Ke-{d} ({split_name}):",
            options=available_muscles,
            default=default_muscles,
            key=f"day_input_{d}",
        )

        if selected_muscles:
            custom_schedule[f"Hari Ke-{d} ({split_name})"] = selected_muscles

    btn_generate = st.button(
        "Ambil Rekomendasi Latihan", type="primary", use_container_width=True
    )


# --- KOLOM KANAN: HASIL PEMPROSESAN JADWAL (HYBRID SYSTEM ENGINE) ---
with col_content:
    st.header("Hasil Rekomendasi Program Latihan")

    if btn_generate:
        if not custom_schedule:
            st.warning(
                "⚠️ Harap pilih minimal satu kelompok otot fokus pada salah satu hari latihan Anda."
            )
        else:
            data_filtered = df.copy()

            # 1. FILTERING: Level
            allowed_levels = allowed_level_map.get(user_level, [user_level])
            data_filtered = data_filtered[
                data_filtered["Level"].isin(allowed_levels)
            ].copy()

            # 2. FILTERING: Lokasi & Alat (Otomatis tanpa input tambahan dari user)
            # 2. FILTERING: Lokasi & Alat
            if user_location == "Gym":
                equipment_candidates = data_filtered["Equipment"].unique().tolist()
            else:
                if has_equipment:
                    equipment_candidates = location_equipment_map["rumah_ada_alat"]
                else:
                    equipment_candidates = location_equipment_map["rumah_tanpa_alat"]

            data_filtered = data_filtered[
                data_filtered["Equipment"].isin(equipment_candidates)
            ].copy()

            # 3. FILTERING: Goal / Tipe
            preferred_types = goal_map[user_goal]["preferred_types"]
            data_filtered = data_filtered[
                data_filtered["Type"].isin(preferred_types)
            ].copy()

            if len(data_filtered) == 0:
                st.error(
                    "❌ Mohon maaf, tidak ada data gerakan di dataset yang mampu memenuhi kriteria profil Anda. Silakan longgarkan pilihan lokasi atau level."
                )
            else:
                # 4. PREDIKSI MACHINE LEARNING
                pred_input = data_filtered[["text", "Type", "Equipment", "Level"]]
                proba = model.predict_proba(pred_input)
                classes = model.classes_
                proba_df = pd.DataFrame(
                    proba, columns=classes, index=data_filtered.index
                )
                data_filtered = pd.concat([data_filtered, proba_df], axis=1)

                # Fungsi Dinamis Pembobotan Aturan Bisnis (Rule-based Scoring)
                def get_rule_score(row):
                    score = 0.5  # Base score
                    if row["Level"] == user_level:
                        score += 0.2
                    if user_location == "Gym" and row["Equipment"] in [
                        "Barbell",
                        "Dumbbell",
                        "Machine",
                        "Cable",
                    ]:
                        score += 0.2
                    elif user_location == "Rumah" and row["Equipment"] in [
                        "Body Only",
                        "Bands",
                    ]:
                        score += 0.15
                    if user_goal == "Muscle Gain" and row["Type"] == "Strength":
                        score += 0.15
                    elif user_goal == "Fat Loss" and row["Type"] in [
                        "Cardio",
                        "Plyometrics",
                    ]:
                        score += 0.15
                    return min(score, 1.0)

                # 5. PENYUSUNAN JADWAL (HYBRID SCORING & SAMPLING)
                used_titles = set()
                pdf_payload = []

                for day_name, target_parts in custom_schedule.items():
                    num_muscles = len(target_parts)
                    base_exercises = exercises_per_day // num_muscles
                    remainder = exercises_per_day % num_muscles
                    day_chunks = []

                    for i, muscle in enumerate(target_parts):
                        quota = base_exercises + (1 if i < remainder else 0)
                        if quota == 0:
                            continue

                        temp = data_filtered[data_filtered["BodyPart"] == muscle].copy()
                        temp = temp[~temp["Title"].isin(used_titles)]
                        if len(temp) == 0:
                            continue

                        # ML Score (70%)
                        if muscle in temp.columns:
                            temp["score_ml"] = temp[[muscle]].iloc[:, 0]
                        else:
                            temp["score_ml"] = 0.0

                        # Rule Score (30%)
                        temp["score_rule"] = temp.apply(get_rule_score, axis=1)

                        # Kalkulasi Skor Akhir
                        temp["final_score"] = (0.7 * temp["score_ml"]) + (
                            0.3 * temp["score_rule"]
                        )
                        temp = temp.sort_values(["final_score"], ascending=False)

                        # Top-K Random Sampling agar hasil bervariasi setiap kali dijalankan
                        top_k_pool = temp.head(10)
                        if len(top_k_pool) > quota:
                            muscle_chunk = top_k_pool.sample(n=quota).copy()
                        else:
                            muscle_chunk = top_k_pool.copy()

                        # Urutkan kembali hasil sampling dari skor tertinggi ke terendah
                        muscle_chunk = muscle_chunk.sort_values(
                            ["final_score"], ascending=False
                        )

                        if len(muscle_chunk) > 0:
                            used_titles.update(muscle_chunk["Title"].tolist())
                            day_chunks.append(muscle_chunk)

                    fokus_teks = ", ".join(target_parts)
                    with st.expander(
                        f"📌 {day_name.upper()} — (Fokus Otot: {fokus_teks})",
                        expanded=True,
                    ):
                        if day_chunks:
                            day_exercises = pd.concat(day_chunks).reset_index(drop=True)
                            day_exercises = day_exercises.sort_values(
                                ["final_score"], ascending=False
                            ).reset_index(drop=True)

                            st.markdown(
                                f"**Panduan Porsi:** Lakukan sebanyak `{goal_map[user_goal]['sets']}` dengan `{goal_map[user_goal]['reps']}` per gerakan."
                            )

                            df_web_display = day_exercises[
                                [
                                    "Title",
                                    "BodyPart",
                                    "Equipment",
                                    "Level",
                                    "Type",
                                    "final_score",
                                ]
                            ].copy()
                            df_web_display["final_score"] = df_web_display[
                                "final_score"
                            ].round(4)

                            st.dataframe(df_web_display, use_container_width=True)

                            pdf_payload.append(
                                {
                                    "day": day_name,
                                    "focus": fokus_teks,
                                    "dataframe": df_web_display,
                                }
                            )
                        else:
                            st.caption(
                                "⚠️ Kuota variasi data gerakan yang aman untuk target otot ini sudah habis."
                            )

                st.success("🎉 Berhasil!")

                if pdf_payload:
                    st.divider()
                    meta_profile = {
                        "level": user_level,
                        "goal": user_goal,
                        "location": user_location,
                        "equipment": (
                            "Semua Alat Gym"
                            if user_location == "Gym"
                            else (
                                "Alat Standard Rumah"
                                if has_equipment
                                else "Berat Badan (Bodyweight)"
                            )
                        ),
                        "volume": f"{goal_map[user_goal]['sets']} x {goal_map[user_goal]['reps']}",
                    }
                    output_pdf_binary = generate_pdf(meta_profile, pdf_payload)

                    st.download_button(
                        label="Unduh Hasil Jadwal Latihan (PDF)",
                        data=bytes(output_pdf_binary),
                        file_name="Rekomendasi Gerakan dan Jadwal.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
    else:
        st.info(
            "Hubungkan parameter profil latihan Anda pada form di bilah kiri, kemudian klik tombol **Ambil Rekomendasi Latihan**."
        )
