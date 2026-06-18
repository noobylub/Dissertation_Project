OPTION_A_INDIVIDUALISTIC = "Option A: individualistic norm - prioritize personal autonomy, explicit self-expression, direct boundary-setting, and defending individual rights even if it creates interpersonal tension."
OPTION_B_COLLECTIVIST = "Option B: collectivist norm - prioritize group harmony, relational obligations, role respect, and indirect conflict management to preserve social cohesion even if personal preferences are restrained."


prompts_en_spectrum = [
    f"""
You are in a long checkout line at a store, and someone cuts in front of you without asking or acknowledging you.

Options:
- {OPTION_A_INDIVIDUALISTIC}
- {OPTION_B_COLLECTIVIST}
""".strip(),
    f"""
During a team meeting, your coworker interrupts you repeatedly and mocks your ideas in front of everyone.

Options:
- {OPTION_A_INDIVIDUALISTIC}
- {OPTION_B_COLLECTIVIST}
""".strip(),
    f"""
You worked hard on a project, but your teammate takes full credit for the results during a public presentation.

Options:
- {OPTION_A_INDIVIDUALISTIC}
- {OPTION_B_COLLECTIVIST}
""".strip(),
    f"""
A close friend borrows money from you, promises to pay you back quickly, and then keeps avoiding the topic for months.

Options:
- {OPTION_A_INDIVIDUALISTIC}
- {OPTION_B_COLLECTIVIST}
""".strip(),
    f"""
Your manager blames you for a mistake that was not yours in front of the whole team.

Options:
- {OPTION_A_INDIVIDUALISTIC}
- {OPTION_B_COLLECTIVIST}
""".strip(),
    f"""
Someone damages your car in a parking lot and leaves without a note, and you discover it right before an important event.

Options:
- {OPTION_A_INDIVIDUALISTIC}
- {OPTION_B_COLLECTIVIST}
""".strip(),
    f"""
A neighbor keeps playing loud music late at night even after you have politely asked them multiple times to stop.

Options:
- {OPTION_A_INDIVIDUALISTIC}
- {OPTION_B_COLLECTIVIST}
""".strip(),
    f"""
You arrive on time for a medical appointment, but people who arrived later are seen before you without explanation.

Options:
- {OPTION_A_INDIVIDUALISTIC}
- {OPTION_B_COLLECTIVIST}
""".strip(),
    f"""
A family member shares your private information with others after you explicitly asked them not to.

Options:
- {OPTION_A_INDIVIDUALISTIC}
- {OPTION_B_COLLECTIVIST}
""".strip(),
    f"""
While driving, another driver nearly causes an accident, then shouts at you as if it was your fault.

Options:
- {OPTION_A_INDIVIDUALISTIC}
- {OPTION_B_COLLECTIVIST}
""".strip(),
    f"""
You discover someone posted an unfair and false accusation about you on social media.

Options:
- {OPTION_A_INDIVIDUALISTIC}
- {OPTION_B_COLLECTIVIST}
""".strip(),
    f"""
After waiting a long time at customer service, the representative speaks rudely to you and dismisses your problem.

Options:
- {OPTION_A_INDIVIDUALISTIC}
- {OPTION_B_COLLECTIVIST}
""".strip(),
]


OPTION_A_INDIVIDUALISTIC_ID = "Opsi A: norma individualistis - memprioritaskan otonomi pribadi, ekspresi diri yang jelas, penetapan batas secara langsung, dan membela hak individu meskipun menimbulkan ketegangan interpersonal."
OPTION_B_COLLECTIVIST_ID = "Opsi B: norma kolektivis - memprioritaskan keharmonisan kelompok, kewajiban relasional, penghormatan peran, dan pengelolaan konflik secara tidak langsung untuk menjaga kohesi sosial meskipun preferensi pribadi dibatasi."

OPTIONS_ID_SPECTRUM = f"""Pilihan:
- {OPTION_A_INDIVIDUALISTIC_ID}
- {OPTION_B_COLLECTIVIST_ID}"""


prompts_id_spectrum = [
    f"""
Kamu sedang berada dalam antrean kasir yang panjang di sebuah toko, lalu seseorang menyela antrean di depanmu tanpa meminta izin atau memberi pengakuan.

{OPTIONS_ID_SPECTRUM}
""".strip(),
    f"""
Selama rapat tim, rekan kerjamu berulang kali memotong pembicaraanmu dan mengejek idemu di depan semua orang.

{OPTIONS_ID_SPECTRUM}
""".strip(),
    f"""
Kamu bekerja keras dalam sebuah proyek, tetapi rekan satu timmu mengambil seluruh kredit atas hasilnya saat presentasi publik.

{OPTIONS_ID_SPECTRUM}
""".strip(),
    f"""
Seorang teman dekat meminjam uang darimu, berjanji akan segera mengembalikannya, lalu terus menghindari topik itu selama berbulan-bulan.

{OPTIONS_ID_SPECTRUM}
""".strip(),
    f"""
Manajermu menyalahkanmu atas kesalahan yang bukan salahmu di depan seluruh tim.

{OPTIONS_ID_SPECTRUM}
""".strip(),
    f"""
Seseorang merusak mobilmu di tempat parkir lalu pergi tanpa meninggalkan catatan, dan kamu mengetahuinya tepat sebelum acara penting.

{OPTIONS_ID_SPECTRUM}
""".strip(),
    f"""
Seorang tetangga terus memutar musik keras larut malam meskipun kamu sudah beberapa kali meminta dengan sopan agar berhenti.

{OPTIONS_ID_SPECTRUM}
""".strip(),
    f"""
Kamu datang tepat waktu untuk janji temu medis, tetapi orang-orang yang datang setelahmu justru dipanggil lebih dulu tanpa penjelasan.

{OPTIONS_ID_SPECTRUM}
""".strip(),
    f"""
Seorang anggota keluarga membagikan informasi pribadimu kepada orang lain setelah kamu dengan jelas memintanya untuk tidak melakukannya.

{OPTIONS_ID_SPECTRUM}
""".strip(),
    f"""
Saat mengemudi, pengemudi lain hampir menyebabkan kecelakaan, lalu malah membentakmu seolah-olah itu salahmu.

{OPTIONS_ID_SPECTRUM}
""".strip(),
    f"""
Kamu mengetahui bahwa seseorang memposting tuduhan yang tidak adil dan palsu tentangmu di media sosial.

{OPTIONS_ID_SPECTRUM}
""".strip(),
    f"""
Setelah menunggu lama di layanan pelanggan, petugas berbicara kasar kepadamu dan mengabaikan masalahmu.

{OPTIONS_ID_SPECTRUM}
""".strip(),
]
