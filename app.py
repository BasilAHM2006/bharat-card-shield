import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from currencies import COUNTRY_CURRENCIES, country_label, FALLBACK_INR_PER_UNIT, _fetch_inr_rates

st.set_page_config(
    page_title="Bharat Card Shield",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)
FX_RATE = 83.0

PLOT_LAYOUT = dict(
    template="simple_white",
    font=dict(family="Outfit, Inter, sans-serif", color="#1E293B", size=12),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=12, r=12, t=16, b=12),
    hoverlabel=dict(bgcolor="#0A2540", font_size=12, font_family="Outfit"),
)

# ==============================================================
# COMPREHENSIVE MULTI-LINGUAL LOCALIZATION (ALL MENUS)
# ==============================================================
LANGUAGES = {
    "English": "en",
    "தமிழ் (Tamil)": "ta",
    "हिन्दी (Hindi)": "hi",
    "اردو (Urdu)": "ur",
    "മലയാളം (Malayalam)": "ml",
    "తెలుగు (Telugu)": "te"
}

I18N = {
    "en": {
        "title": "Bharat Card Shield",
        "tagline": "Watch a rupee debit card on the right while you inspect unusual spends — kirana swipes, mall bills, and late-night overseas cash.",
        "pill_text": "India rupee · live demo · not a real bank",
        "how_to_read": "<b>How to read this:</b> every payment gets a fraud risk from 0% to 100%. At <b>50% or more</b> the rupee amount on the card is held for an extra OTP check instead of going through.",
        "genuine": "Genuine",
        "fraud": "Fraud",
        
        # Dashboard
        "kpi_reviewed": "PAYMENTS REVIEWED",
        "kpi_flagged": "FLAGGED AS FRAUD",
        "kpi_share": "FRAUD SHARE",
        "kpi_typical": "TYPICAL FRAUD AMOUNT",
        "sub_rows": "Rows in the sample rupee ledger",
        "sub_unauth": "Unauthorised card spends",
        "sub_rare": "Fraud is rare — that makes it hard to catch",
        "sub_avg": "Average value of flagged spends",
        "threat_title": "Vigilance Architecture & Threat Directives",
        "threat_desc": "Financial cyber attacks target structural delay points—predominantly late-night clearing intervals and fast multi-tier fund routing. Pre-settlement screening neutralizes vectors before interbank dispersal occurs.",
        "theory_1_title": "1. Temporal Attack Clustering",
        "theory_1_desc": "Synthetic intrusions cluster heavily between 11:00 PM and 5:00 AM (IST) to exploit delayed cardholder SMS confirmation and lower overnight bank staff coverage.",
        "theory_2_title": "2. Extreme Class Imbalance & PCA Scoring",
        "theory_2_desc": "With illicit transfers making up less than 0.20% of traffic, gross accuracy is uninformative. Sentinel relies on high Recall calibration and principal component deviations to isolate zero-day threats.",
        "dir_title": "OFFICIAL NATIONAL THREAT DIRECTORIES",
        "chart_time_title": "When do fraud attempts happen?",
        "chart_time_sub": "Hours run from midnight (0) to 11 p.m. (23). Fraud often clusters after late-night hours.",
        "chart_pie_title": "Normal vs flagged",
        "chart_pie_sub": "Almost every rupee spend is genuine. The model must still find the few that are not.",

        # How It Works
        "hiw_kicker": "Guide",
        "hiw_title": "How computers catch card fraud",
        "hiw_sub": "Banks used to rely on fixed rupee cut-offs. This system looks at many clues at once, then scores how unusual a spend looks.",
        "hiw_p1_title": "1. Why simple rules fall short",
        "hiw_p1_body": "Old systems said: “Block foreign payment over ₹4,00,000.” That causes false alarms and lets attackers bypass thresholds.",
        "hiw_p2_title": "2. Fraud is rare (and that is a problem)",
        "hiw_p2_body": "In real networks, fraud is only 0.1%–0.2%. A model guessing 'always safe' has 99.8% accuracy but catches zero fraud.",
        "hiw_p3_title": "3. Making missed fraud more expensive",
        "hiw_p3_body": "Missing fraud is penalized 50 times heavier than raising a false alarm.",
        "hiw_p4_title": "4. Catching fraud vs raising alarms",
        "hiw_p4_body": "High Recall is prioritized to catch actual theft over minimizing OTP challenges.",

        # Spending patterns
        "sp_kicker": "CHARTS",
        "sp_title": "Compare genuine spending with fraud",
        "sp_sub": "Pick a detail — rupee amount, hour, device trust, or a behavior score — and see how flagged payments differ from normal ones.",
        "sp_compare_label": "What do you want to compare?",
        "sp_bars_label": "Chart detail (number of bars)",
        "sp_box_title": "Payment size in ₹ (log scale)",
        "sp_box_caption": "The log scale makes both small and large values visible together.",
        "sp_corr_title": "Do these details move together?",
        "sp_corr_caption": "Darker blue indicates features rising and falling together.",

        # Check a payment
        "chk_kicker": "LIVE CHECK",
        "chk_title": "Test a rupee payment before it is sent",
        "chk_sub": "Pick a ready-made story or type an amount in ₹. The card updates live.",
        "chk_stories": "Start from a common Indian story",
        "btn_kirana": "Kirana shop · ₹3,735",
        "btn_atm": "3 AM overseas ATM · ₹51,460",
        "btn_mall": "Mumbai mall · ₹1,03,750",
        "btn_reset": "Reset to default",
        "chk_details": "Payment details",
        "lbl_amount": "Amount (₹)",
        "lbl_hour": "Hour of the payment",
        "lbl_loc": "Where was it made?",
        "opt_india": "Inside India",
        "opt_intl": "Outside India (international)",
        "lbl_trust": "Device trust (0 = unknown phone, 1 = known phone)",
        "btn_run": "Check this payment",
        "chk_result": "Result",
        "msg_prompt": "Fill in the rupee amount on the left, then press Check this payment.",

        # About system
        "about_kicker": "Project",
        "about_title": "What this demo is",
        "about_sub": "An industry-style machine-learning project detecting anomalous card payments.",
        "about_p1_title": "In one sentence",
        "about_p1_body": "Bharat Card Shield inspects 30,000 ledger records using a 120-tree Random Forest.",
        "about_p2_title": "How the pipeline is built",
        "about_p2_body": "Sample data undergoes class balancing, behavioral feature isolation (V1–V4), and real-time inference.",

        # Accuracy
        "acc_kicker": "Scoreboard",
        "acc_title": "Did the model actually catch fraud?",
        "acc_sub": "These numbers come from a held-out test set unseen during training.",
        "acc_recall": "Frauds caught (recall)",
        "acc_prec": "Alarm quality (precision)",
        "acc_cm_title": "Confusion matrix (all sample rows)",
        "acc_imp_title": "Which clues the model leans on",
    },
    "ta": {
        "title": "பாரத் கார்டு ஷீல்டு",
        "tagline": "கிரானா கடைகள், வணிக வளாகங்கள் மற்றும் நள்ளிரவு வெளிநாட்டு பண பரிவர்த்தனைகளை ஆய்வு செய்யும் போது வலதுபுறத்தில் உள்ள ரூபாய் டெபிட் கார்டைக் கவனியுங்கள்.",
        "pill_text": "இந்திய ரூபாய் · நேரடி டெமோ · மாதிரி வங்கி மட்டுமே",
        "how_to_read": "<b>விளக்கம்:</b> ஒவ்வொரு பரிவர்த்தனைக்கும் 0% முதல் 100% வரை ஆபத்து கணக்கிடப்படுகிறது. <b>50% அல்லது அதற்கு மேல்</b> இருந்தால் பரிவர்த்தனை நிறுத்தப்பட்டு கூடுதல் OTP கேட்கப்படும்.",
        "genuine": "சாதாரண பரிவர்த்தனை",
        "fraud": "மோசடி என குறிக்கப்பட்டது",
        
        # Dashboard
        "kpi_reviewed": "ஆய்வு செய்யப்பட்ட பரிவர்த்தனைகள்",
        "kpi_flagged": "மோசடியாகக் குறிக்கப்பட்டவை",
        "kpi_share": "மோசடி சதவீதம்",
        "kpi_typical": "சராசரி மோசடி தொகை",
        "sub_rows": "மாதிரி ரூபாய் பதிவேட்டில் உள்ள பதிவுகள்",
        "sub_unauth": "அங்கீகரிக்கப்படாத கார்டு செலவுகள்",
        "sub_rare": "மோசடி மிகவும் அரிதானது — கண்டறிவது கடினம்",
        "sub_avg": "அடையாளம் காணப்பட்ட சராசரி மதிப்பு",
        "threat_title": "பாதுகாப்பு கட்டமைப்பு மற்றும் வழிகாட்டுதல்கள்",
        "threat_desc": "நள்ளிரவு வங்கி பரிவர்த்தனை தாமதங்களை குறிவைத்து நிதி மோசடிகள் நடைபெறுகின்றன. பணம் மாற்றப்படுவதற்கு முன்பே செயற்கை நுண்ணறிவு இதனைத் தடுக்கிறது.",
        "theory_1_title": "1. நள்ளிரவு தாக்குதல் குவியம்",
        "theory_1_desc": "இரவு 11:00 மணி முதல் காலை 5:00 மணி வரை வாடிக்கையாளர்கள் விழிப்புடன் இல்லாத நேரத்தை பயன்படுத்தி மோசடிகள் அதிகம் நடக்கின்றன.",
        "theory_2_title": "2. தரவு சமநிலையின்மை மற்றும் PCA மதிப்பீடு",
        "theory_2_desc": "மொத்த பரிவர்த்தனைகளில் மோசடி 0.20% க்கும் குறைவு என்பதால், துல்லியத்தை விட ரீகால் (Recall) மற்றும் நடத்தை முரண்பாடுகளுக்கே முன்னுரிமை அளிக்கப்படுகிறது.",
        "dir_title": "அதிகாரப்பூர்வ தேசிய பாதுகாப்பு தளங்கள்",
        "chart_time_title": "மோசடி முயற்சிகள் எப்போது நிகழ்கின்றன?",
        "chart_time_sub": "நள்ளிரவு (0) முதல் இரவு 11 மணி (23) வரை. நள்ளிரவில் மோசடிகள் அதிகம் பதிவாகின்றன.",
        "chart_pie_title": "சாதாரண vs மோசடி பரிவர்த்தனைகள்",
        "chart_pie_sub": "பெரும்பாலான பரிவர்த்தனைகள் உண்மையானவை. சில மோசடிகளை மாதிரி கண்டறிய வேண்டும்.",

        # How It Works
        "hiw_kicker": "வழிகாட்டி",
        "hiw_title": "கணினிகள் எவ்வாறு மோசடியைக் கண்டறிகின்றன?",
        "hiw_sub": "நிலையான வரம்புகளுக்குப் பதிலாக பல காரணிகளை ஆராய்ந்து கணினி முடிவெடுக்கிறது.",
        "hiw_p1_title": "1. எளிய விதிகள் ஏன் தோல்வியடைகின்றன",
        "hiw_p1_body": "பழைய விதிகளில் வெளிநாட்டு பரிவர்த்தனைகளைத் தடுப்பது போன்ற கட்டுப்பாடுகள் எளிதில் ஏமாற்றப்படுகின்றன.",
        "hiw_p2_title": "2. மோசடி மிகவும் அரிதானது",
        "hiw_p2_body": "மொத்த பரிவர்த்தனைகளில் 0.2% மட்டுமே மோசடி என்பதால் பொதுவான துல்லியம் பயனற்றது.",
        "hiw_p3_title": "3. மோசடியை தவறவிடுவது அதிக இழப்பைத் தரும்",
        "hiw_p3_body": "மோசடியைத் தவறவிடுவது சாதாரண அலாரத்தை விட 50 மடங்கு தீவிரமாகக் கருதப்படுகிறது.",
        "hiw_p4_title": "4. மோசடியைக் கண்டறிதல் vs வாடிக்கையாளர் திருப்தி",
        "hiw_p4_body": "தவறான அலாரங்களை விட திருட்டைத் தடுப்பதற்கே (Recall) முதலிடம் வழங்கப்படுகிறது.",

        # Spending patterns
        "sp_kicker": "வரைபடங்கள்",
        "sp_title": "சாதாரண மற்றும் மோசடி செலவுகளை ஒப்பிடுக",
        "sp_sub": "தொகை, நேரம், சாதன நம்பிக்கை அல்லது நடத்தை மதிப்பெண் விவரங்களைத் தேர்ந்தெடுத்து வேறுபாடுகளைக் காண்க.",
        "sp_compare_label": "நீங்கள் எதை ஒப்பிட விரும்புகிறீர்கள்?",
        "sp_bars_label": "வரைபட விவரம் (பார்களின் எண்ணிக்கை)",
        "sp_box_title": "செலவுத் தொகை ₹ இல் (லாக் அளவுகோல்)",
        "sp_box_caption": "சிறிய மற்றும் பெரிய தொகைகளை ஒரே வரைபடத்தில் பார்க்க உதவுகிறது.",
        "sp_corr_title": "இந்த விவரங்கள் ஒன்றோடொன்று தொடர்புடையவையா?",
        "sp_corr_caption": "அடர் நீலம் இரு விவரங்களும் ஒன்றாக நகர்வதைக் குறிக்கிறது.",

        # Check a payment
        "chk_kicker": "நேரடி சோதனை",
        "chk_title": "பணம் செலுத்தும் முன் பரிவர்த்தனையை சோதிக்கவும்",
        "chk_sub": "மாதிரி விவரத்தை தேர்ந்தெடுக்கவும் அல்லது தொகையை உள்ளிட்டு ஆபத்து மதிப்பெண்ணைக் காணவும்.",
        "chk_stories": "மாதிரிப் பரிவர்த்தனைகள்",
        "btn_kirana": "கிரானா கடை · ₹3,735",
        "btn_atm": "நள்ளிரவு 3 மணி ஏடிஎம் · ₹51,460",
        "btn_mall": "வணிக வளாகம் · ₹1,03,750",
        "btn_reset": "மீட்டமைக்கவும்",
        "chk_details": "பரிவர்த்தனை விவரங்கள்",
        "lbl_amount": "தொகை (₹)",
        "lbl_hour": "பரிவர்த்தனை நேரம்",
        "lbl_loc": "எங்கே செலுத்தப்பட்டது?",
        "opt_india": "இந்தியாவிற்குள்",
        "opt_intl": "வெளிநாட்டில் (சர்வதேசம்)",
        "lbl_trust": "சாதன நம்பிக்கை",
        "btn_run": "பரிவர்த்தனையை சரிபார்க்கவும்",
        "chk_result": "முடிவு",
        "msg_prompt": "விவரங்களை நிரப்பி 'பரிவர்த்தனையை சரிபார்க்கவும்' பொத்தானை அழுத்தவும்.",

        # About system
        "about_kicker": "திட்டம்",
        "about_title": "இந்த டெமோ என்ன செய்கிறது?",
        "about_sub": "அசாதாரண கார்டு பரிவர்த்தனைகளைக் கண்டறியும் மெஷின் லேர்னிங் திட்டம்.",
        "about_p1_title": "ஒரே வரியில்",
        "about_p1_body": "பாரத் கார்டு ஷீல்டு என்பது 30,000 பரிவர்த்தனைகளை ஆய்வு செய்து மோசடிகளை அடையாளம் காணும் அமைப்பு.",
        "about_p2_title": "கட்டமைப்பு எவ்வாறு உருவாக்கப்பட்டது",
        "about_p2_body": "மாதிரி தரவு 120 மரங்கள் கொண்ட ரேண்டம் பாரஸ்ட் மாடல் மூலம் பயிற்றுவிக்கப்பட்டுள்ளது.",

        # Accuracy
        "acc_kicker": "மதிப்பெண் பலகை",
        "acc_title": "மாதிரி எவ்வாறு செயல்பட்டது?",
        "acc_sub": "பயிற்சியில் ஈடுபடுத்தப்படாத சோதனைத் தரவிலிருந்து பெறப்பட்ட முடிவுகள்.",
        "acc_recall": "கண்டறியப்பட்ட மோசடிகள் (Recall)",
        "acc_prec": "எச்சரிக்கைத் தரம் (Precision)",
        "acc_cm_title": "குழப்ப அணி (Confusion Matrix)",
        "acc_imp_title": "முக்கியக் காரணிகள்",
    },
    "hi": {
        "title": "भारत कार्ड शील्ड",
        "tagline": "किराना स्टोर, मॉल बिल और देर रात विदेशी लेन-देन की जांच करते समय दाईं ओर दिए गए रुपिया डेबिट कार्ड पर नज़र रखें।",
        "pill_text": "भारतीय रुपया · लाइव डेमो · कोई वास्तविक बैंक नहीं",
        "how_to_read": "<b>निर्देश:</b> प्रत्येक भुगतान को 0% से 100% तक जोखिम स्कोर मिलता है। <b>50% या अधिक</b> होने पर अतिरिक्त ओटीपी जांच के लिए लेन-देन रोक दिया जाता है।",
        "genuine": "वैध लेन-देन",
        "fraud": "संदिग्ध धोखाधड़ी",
        
        # Dashboard
        "kpi_reviewed": "समीक्षित भुगतान",
        "kpi_flagged": "संदिग्ध धोखाधड़ी",
        "kpi_share": "धोखाधड़ी अनुपात",
        "kpi_typical": "औसत धोखाधड़ी राशि",
        "sub_rows": "नमूना खाता बही में कुल लेन-देन",
        "sub_unauth": "अनधिकृत कार्ड लेन-देन",
        "sub_rare": "धोखाधड़ी दुर्लभ है — पकड़ना कठिन है",
        "sub_avg": "संदिग्ध भुगतानों का औसत मूल्य",
        "threat_title": "सतर्कता वास्तुकला एवं सुरक्षा निर्देश",
        "threat_desc": "वित्तीय साइबर हमले मुख्य रूप से देर रात के समय को निशाना बनाते हैं। एआई निपटान से पहले ही इसे निष्प्रभावी कर देता है।",
        "theory_1_title": "1. रात्रि कालीन जोखिम क्लस्टर",
        "theory_1_desc": "हमलावर रात 11:00 बजे से सुबह 5:00 बजे के बीच अनधिकृत स्वाइप करते हैं।",
        "theory_2_title": "2. अत्यधिक वर्ग असंतुलन और PCA स्कोरिंग",
        "theory_2_desc": "धोखाधड़ी 0.20% से भी कम होने के कारण मॉडल रिकॉल पर निर्भर करता है।",
        "dir_title": "आधिकारिक राष्ट्रीय सुरक्षा पोर्टल",
        "chart_time_title": "धोखाधड़ी के प्रयास कब होते हैं?",
        "chart_time_sub": "घंटे आधी रात (0) से रात 11 बजे (23) तक।",
        "chart_pie_title": "सामान्य बनाम संदिग्ध लेन-देन",
        "chart_pie_sub": "अधिकांश भुगतान वैध होते हैं।",

        # How It Works
        "hiw_kicker": "गाइड",
        "hiw_title": "कंप्यूटर कार्ड धोखाधड़ी को कैसे पकड़ते हैं?",
        "hiw_sub": "नियमों के बजाय बहुआयामी व्यवहार का विश्लेषण किया जाता है।",
        "hiw_p1_title": "1. सरल नियम क्यों विफल होते हैं",
        "hiw_p1_body": "निश्चित सीमा वाले नियम आसानी से तोड़े जा सकते हैं।",
        "hiw_p2_title": "2. धोखाधड़ी अत्यंत दुर्लभ है",
        "hiw_p2_body": "0.2% से कम धोखाधड़ी होने पर साधारण एक्यूरेसी अर्थहीन होती है।",
        "hiw_p3_title": "3. छूटी हुई धोखाधड़ी अधिक महंगी होती है",
        "hiw_p3_body": "धोखाधड़ी छूटने पर 50 गुना अधिक दंड निर्धारित है।",
        "hiw_p4_title": "4. रिकॉल बनाम प्रिसिजन",
        "hiw_p4_body": "चोरी रोकने के लिए उच्च रिकॉल (Recall) आवश्यक है।",

        # Spending patterns
        "sp_kicker": "चार्ट",
        "sp_title": "वैध खर्च और धोखाधड़ी की तुलना करें",
        "sp_sub": "कोई विवरण चुनें और देखें कि संदिग्ध भुगतान सामान्य से कैसे भिन्न हैं।",
        "sp_compare_label": "आप किसकी तुलना करना चाहते हैं?",
        "sp_bars_label": "चार्ट विवरण (सलाखों की संख्या)",
        "sp_box_title": "₹ में भुगतान का आकार (लॉग स्केल)",
        "sp_box_caption": "छोटे और बड़े दोनों मानों को एक साथ देखना संभव बनाता है।",
        "sp_corr_title": "क्या ये विवरण एक साथ बदलते हैं?",
        "sp_corr_caption": "गहरा नीला रंग दोनों स्तंभों के संबंध को दर्शाता है।",

        # Check a payment
        "chk_kicker": "लाइव जांच",
        "chk_title": "रुपया भेजने से पहले भुगतान की जांच करें",
        "chk_sub": "तैयार उदाहरण चुनें या ₹ में राशि लिखकर स्कोर देखें।",
        "chk_stories": "सामान्य उदाहरण",
        "btn_kirana": "किराना स्टोर · ₹3,735",
        "btn_atm": "देर रात 3 बजे एटीएम · ₹51,460",
        "btn_mall": "मुंबई मॉल · ₹1,03,750",
        "btn_reset": "रीसेट करें",
        "chk_details": "भुगतान विवरण",
        "lbl_amount": "राशि (₹)",
        "lbl_hour": "भुगतान का समय",
        "lbl_loc": "भुगतान कहाँ किया गया?",
        "opt_india": "भारत के अंदर",
        "opt_intl": "विदेश में",
        "lbl_trust": "डिवाइस ट्रस्ट",
        "btn_run": "इस भुगतान की जांच करें",
        "chk_result": "परिणाम",
        "msg_prompt": "विवरण भरें और 'इस भुगतान की जांच करें' बटन दबाएं।",

        # About system
        "about_kicker": "परियोजना",
        "about_title": "यह डेमो क्या है?",
        "about_sub": "मशीन-लर्निंग आधारित कार्ड सुरक्षा परियोजना।",
        "about_p1_title": "एक वाक्य में",
        "about_p1_body": "भारत कार्ड शील्ड 30,000 लेन-देन का विश्लेषण करने वाला मॉडल है।",
        "about_p2_title": "पाइपलाइन कैसे बनाई गई है",
        "about_p2_body": "रैंडम फॉरेस्ट मॉडल 120 ट्रीज के साथ प्रशिक्षित किया गया है।",

        # Accuracy
        "acc_kicker": "स्कोरबोर्ड",
        "acc_title": "मॉडल की सटीकता",
        "acc_sub": "परीक्षण डेटा से प्राप्त परिणाम।",
        "acc_recall": "पकड़ी गई धोखाधड़ी (Recall)",
        "acc_prec": "अलार्म गुणवत्ता (Precision)",
        "acc_cm_title": "कन्फ्यूजन मैट्रिक्स",
        "acc_imp_title": "महत्वपूर्ण कारक",
    },
    "ur": {
        "title": "بھارت کارڈ شیلڈ",
        "tagline": "کرانہ، شاپنگ مال اور رات گئے غیر ملکی ادائیگیاں چیک کرتے وقت دائیں جانب موجود روپے ڈیبٹ کارڈ کا معائنہ کریں۔",
        "pill_text": "ہندوستانی روپیہ · لائیو ڈیمو · فرضی بینک",
        "how_to_read": "<b>طریقہ کار:</b> ہر ادائیگی پر 0% سے 100% رسک اسکور ملتا ہے۔ <b>50% یا زائد</b> پر رقم روکی جاتی ہے۔",
        "genuine": "اصلی ادائیگی",
        "fraud": "مشکوک فراڈ",
        
        "kpi_reviewed": "جانچے گئے ادائیگیاں",
        "kpi_flagged": "مشکوک فراڈ",
        "kpi_share": "فراڈ کا تناسب",
        "kpi_typical": "اوسط فراڈ کی رقم",
        "sub_rows": "سیمپل لیجر میں کل اندراجات",
        "sub_unauth": "غیر مجاز کارڈ کے اخراجات",
        "sub_rare": "فراڈ شاذ و نادر ہوتا ہے",
        "sub_avg": "مشکوک ٹرانزیکشنز کی اوسط قیمت",
        "threat_title": "حفاظتی ڈھانچہ اور انتباہی ہدایات",
        "threat_desc": "مالیاتی سائبر حملے رات کے وقت کلیئرنگ کے وقفوں کو نشانہ بناتے ہیں۔",
        "theory_1_title": "1. رات کے وقت حملوں کا ارتکاز",
        "theory_1_desc": "حملہ آور رات 11 بجے سے صبح 5 بجے کے درمیان چوری کرتے ہیں۔",
        "theory_2_title": "2. ڈیٹا کا عدم توازن اور PCA اسکورنگ",
        "theory_2_desc": "ماڈل درستگی کے بجائے غیر معمولی رویوں کی شناخت پر انحصار کرتا ہے۔",
        "dir_title": "سرکاری قومی سائبر دفاعی پورٹلز",
        "chart_time_title": "فراڈ کی کوششیں کب ہوتی ہیں؟",
        "chart_time_sub": "آدھی رات سے رات 11 بجے تک۔",
        "chart_pie_title": "عام بمقابلہ مشکوک ادائیگیاں",
        "chart_pie_sub": "زیادہ تر ادائیگیاں اصلی ہوتی ہیں۔",

        "hiw_kicker": "رہنما",
        "hiw_title": "کمپیوٹرز فراڈ کیسے پکڑتے ہیں؟",
        "hiw_sub": "پرانے کٹوتی اصولوں کے بجائے جامع ڈیٹا تجزیہ کیا جاتا ہے۔",
        "hiw_p1_title": "1. پرانے اصول کیوں ناکام ہوتے ہیں",
        "hiw_p1_body": "جامد حدود والے قوانین کو حملہ آور آسانی سے چکمہ دے دیتے ہیں۔",
        "hiw_p2_title": "2. فراڈ انتہائی نایاب ہے",
        "hiw_p2_body": "صرف 0.2% فراڈ ہونے پر روایتی ایکوریسی بے معنی ہوتی ہے۔",
        "hiw_p3_title": "3. فراڈ کا چھوٹ جانا مہنگا پڑتا ہے",
        "hiw_p3_body": "فراڈ چھوڑنے پر ماڈل کو 50 گنا زیادہ جزا و سزا دی جاتی ہے۔",
        "hiw_p4_title": "4. ریکال بمقابلہ پریسیشن",
        "hiw_p4_body": "چوری پکڑنا (Recall) پہلی ترجیح ہے۔",

        "sp_kicker": "چارٹس",
        "sp_title": "اصلی اور مشکوک اخراجات کا موازنہ کریں",
        "sp_sub": "تفصیل منتخب کریں اور مشکوک و عام ادائیگیاں پرکھیں۔",
        "sp_compare_label": "آپ کس چیز کا موازنہ کرنا چاہتے ہیں؟",
        "sp_bars_label": "بارز کی تعداد",
        "sp_box_title": "ادائیگی کا حجم روپے میں (لاگ اسکیل)",
        "sp_box_caption": "چھوٹی اور بڑی رقوم ایک ساتھ دکھاتا ہے۔",
        "sp_corr_title": "کیا یہ خصوصیات آپس میں جڑی ہیں؟",
        "sp_corr_caption": "گہرا نیلا رنگ باہمی تعلق کو ظاہر کرتا ہے۔",

        "chk_kicker": "لائیو چیک",
        "chk_title": "رقم بھیجنے سے پہلے ادائیگی کی جانچ کریں",
        "chk_sub": "مثال منتخب کریں یا رقم لکھ کر چیک کریں۔",
        "chk_stories": "عام ادائیگیوں کی مثالیں",
        "btn_kirana": "کرانہ دکان · ₹3,735",
        "btn_atm": "رات 3 بجے اے ٹی ایم · ₹51,460",
        "btn_mall": "ممبئی مال · ₹1,03,750",
        "btn_reset": "ری سیٹ کریں",
        "chk_details": "ادائیگی کی تفصیلات",
        "lbl_amount": "رقم (₹)",
        "lbl_hour": "ادائیگی کا وقت",
        "lbl_loc": "ادائیگی کہاں ہوئی؟",
        "opt_india": "ہندوستان کے اندر",
        "opt_intl": "بیرون ملک",
        "lbl_trust": "ڈیوائس کا اعتبار",
        "btn_run": "اس ادائیگی کو چیک کریں",
        "chk_result": "نتیجہ",
        "msg_prompt": "تفصیلات درج کر کے چیک بٹن دبائیں۔",

        "about_kicker": "پروجیکٹ",
        "about_title": "یہ ڈیمو کیا ہے؟",
        "about_sub": "مشین لرننگ کارڈ سیکیورٹی پلیٹ فارم۔",
        "about_p1_title": "ایک جملے میں",
        "about_p1_body": "30,000 نمونہ ادائیگیوں کو اسکور کرنے والا ماڈل۔",
        "about_p2_title": "سسٹم کیسے بنا ہے",
        "about_p2_body": "120 ٹریز کے ساتھ رینڈم فارسٹ الگورتھم۔",

        "acc_kicker": "اسکور کارڈ",
        "acc_title": "ماڈل کی کارکردگی",
        "acc_sub": "الگ ٹیسٹ ڈیٹا کے نتائج۔",
        "acc_recall": "پکڑے گئے فراڈ (Recall)",
        "acc_prec": "الارم کی کوالٹی (Precision)",
        "acc_cm_title": "کنفیوژن میٹرکس",
        "acc_imp_title": "اہم خصوصیات",
    },
    "ml": {
        "title": "ഭാരത് കാർഡ് ഷീൽഡ്",
        "tagline": "കടകൾ, മാളുകൾ, അർദ്ധരാത്രിയിലെ വിദേശ ഇടപാടുകൾ എന്നിവ പരിശോധിക്കുമ്പോൾ വലതുവശത്തുള്ള രൂപ ഡെബിറ്റ് കാർഡ് ശ്രദ്ധിക്കുക.",
        "pill_text": "ഇന്ത്യൻ രൂപ · ലൈവ് ഡെമോ · ഡെമോ ബാങ്ക്",
        "how_to_read": "<b>വിശദീകരണം:</b> 0% മുതൽ 100% വരെ റിസ്ക് കണക്കാക്കുന്നു. <b>50% അല്ലെങ്കിൽ കൂടുതൽ</b> ആയാൽ ഒടിപി പരിശോധനയ്ക്കായി തടയുന്നു.",
        "genuine": "യഥാർത്ഥ ഇടപാട്",
        "fraud": "തട്ടിപ്പായി കണ്ടെത്തിയത്",
        
        "kpi_reviewed": "പരിശോധിച്ച ഇടപാടുകൾ",
        "kpi_flagged": "തട്ടിപ്പായി കണ്ടെത്തിയവ",
        "kpi_share": "തട്ടിപ്പിന്റെ ശതമാനം",
        "kpi_typical": "ശരാശരി തട്ടിപ്പ് തുക",
        "sub_rows": "ലെഡ്ജറിലെ ഇടപാടുകൾ",
        "sub_unauth": "അനധികൃത കാർഡ് ഇടപാടുകൾ",
        "sub_rare": "തട്ടിപ്പ് വിരളമാണ്",
        "sub_avg": "കണ്ടെത്തിയ തുകയുടെ ശരാശരി",
        "threat_title": "സുരക്ഷാ ഘടനയും നിർദ്ദേശങ്ങളും",
        "threat_desc": "അർദ്ധരാത്രിയിലെ ബാങ്കിംഗ് ഇടവേളകളെ ലക്ഷ്യമിട്ടാണ് സാമ്പത്തിക തട്ടിപ്പുകൾ നടക്കുന്നത്.",
        "theory_1_title": "1. രാത്രികാല ആക്രമണങ്ങൾ",
        "theory_1_desc": "രാത്രി 11 മുതൽ രാവിലെ 5 വരെയുള്ള സമയം തട്ടിപ്പുകാർ ദുരുപയോഗം ചെയ്യുന്നു.",
        "theory_2_title": "2. ഡാറ്റാ അസന്തുലിതാവസ്ഥയും PCA സ്കോറിംഗും",
        "theory_2_desc": "തട്ടിപ്പുകൾ കുറവായതിനാൽ അസാധാരണ ഇടപാടുകൾ തിരിച്ചറിയുന്നതിലാണ് ശ്രദ്ധ.",
        "dir_title": "ദേശീയ സുരക്ഷാ പോർട്ടലുകൾ",
        "chart_time_title": "തട്ടിപ്പുകൾ എപ്പോഴാണ് നടക്കുന്നത്?",
        "chart_time_sub": "രാത്രി 0 മുതൽ 23 മണി വരെ.",
        "chart_pie_title": "സാധാരണ vs തട്ടിപ്പ് ഇടപാടുകൾ",
        "chart_pie_sub": "മിക്ക ഇടപാടുകളും യഥാർത്ഥമാണ്.",

        "hiw_kicker": "വഴികാട്ടി",
        "hiw_title": "കമ്പ്യൂട്ടറുകൾ തട്ടിപ്പ് എങ്ങനെ കണ്ടെത്തുന്നു?",
        "hiw_sub": "നിയമങ്ങൾക്ക് പകരം വിവിധ ശീലങ്ങൾ വിലയിരുത്തുന്നു.",
        "hiw_p1_title": "1. ലളിതമായ നിയമങ്ങൾ പരാജയപ്പെടുന്നത് എന്തുകൊണ്ട്",
        "hiw_p1_body": "പരിധി അടിസ്ഥാനമാക്കിയുള്ള പഴയ നിയമങ്ങൾ എളുപ്പത്തിൽ ലംഘിക്കപ്പെടുന്നു.",
        "hiw_p2_title": "2. തട്ടിപ്പ് അപൂർവമാണ്",
        "hiw_p2_body": "0.2% മാത്രം തട്ടിപ്പുള്ളതിനാൽ കൃത്യത കൊണ്ട് കാര്യമില്ല.",
        "hiw_p3_title": "3. തട്ടിപ്പ് നഷ്ടപ്പെടുന്നത് ഗുരുതരമാണ്",
        "hiw_p3_body": "തട്ടിപ്പ് വിട്ടുപോകാതിരിക്കാൻ 50 ഇരട്ടി പ്രാധാന്യം നൽകുന്നു.",
        "hiw_p4_title": "4. റീകോൾ പ്രാധാന്യം",
        "hiw_p4_body": "മോഷ്ടിക്കപ്പെടുന്നത് തടയുന്നതിനാണ് (Recall) മുൻഗണന.",

        "sp_kicker": "ചാർട്ടുകൾ",
        "sp_title": "യഥാർത്ഥ ചെലവുകളും തട്ടിപ്പുകളും താരതമ്യം ചെയ്യുക",
        "sp_sub": "വിശദാംശങ്ങൾ തിരഞ്ഞെടുത്ത് മാറ്റങ്ങൾ കാണുക.",
        "sp_compare_label": "താരതമ്യം ചെയ്യേണ്ടത് തിരഞ്ഞെടുക്കുക",
        "sp_bars_label": "ബാറുകളുടെ എണ്ണം",
        "sp_box_title": "ഇടപാട് തുക (ലോഗ് സ്കെയിൽ)",
        "sp_box_caption": "ചെറിയതും വലിയതുമായ തുകകൾ ഒരുമിച്ച് കാണിക്കുന്നു.",
        "sp_corr_title": "പരസ്പര ബന്ധം",
        "sp_corr_caption": "കടും നീല പരസ്പര ബന്ധം വ്യക്തമാക്കുന്നു.",

        "chk_kicker": "തത്സമയ പരിശോധന",
        "chk_title": "പണം അയയ്ക്കുന്നതിന് മുൻപ് പരിശോധിക്കുക",
        "chk_sub": "തുക നൽകി റിസ്ക് സ്കോർ അറിയുക.",
        "chk_stories": "ഉദാഹരണങ്ങൾ",
        "btn_kirana": "പലചരക്ക് കട · ₹3,735",
        "btn_atm": "രാത്രി 3 മണി എടിഎം · ₹51,460",
        "btn_mall": "ഷോപ്പിംഗ് മാൾ · ₹1,03,750",
        "btn_reset": "റീസെറ്റ്",
        "chk_details": "ഇടപാട് വിവരങ്ങൾ",
        "lbl_amount": "തുക (₹)",
        "lbl_hour": "സമയം",
        "lbl_loc": "ഇടപാട് എവിടെയാണ്?",
        "opt_india": "ഇന്ത്യയ്ക്കുള്ളിൽ",
        "opt_intl": "വിദേശത്ത്",
        "lbl_trust": "ഉപകരണ വിശ്വാസ്യത",
        "btn_run": "പരിശോധിക്കുക",
        "chk_result": "ഫലം",
        "msg_prompt": "വിവരങ്ങൾ നൽകിയ ശേഷം പരിശോധിക്കുക ബട്ടൺ അമർത്തുക.",

        "about_kicker": "പ്രോജക്റ്റ്",
        "about_title": "ഈ ഡെമോ എന്താണ്?",
        "about_sub": "മെഷീൻ ലേണിംഗ് കാർഡ് സുരക്ഷാ സംവിധാനം.",
        "about_p1_title": "ഒറ്റ വാക്യത്തിൽ",
        "about_p1_body": "30,000 ഇടപാടുകൾ പരിശോധിച്ച് തട്ടിപ്പുകൾ കണ്ടെത്തുന്ന സംവിധാനം.",
        "about_p2_title": "നിർമ്മാണം",
        "about_p2_body": "120 മരങ്ങളുള്ള റാൻഡം ഫോറസ്റ്റ് മോഡൽ.",

        "acc_kicker": "സ്കോർബോർഡ്",
        "acc_title": "കൃത്യത എത്രമാത്രം?",
        "acc_sub": "ടെസ്റ്റ് ഡാറ്റാബേസിൽ നിന്നുള്ള ഫലങ്ങൾ.",
        "acc_recall": "കണ്ടെത്തിയ തട്ടിപ്പുകൾ (Recall)",
        "acc_prec": "കൃത്യത (Precision)",
        "acc_cm_title": "കൺഫ്യൂഷൻ മാട്രിക്സ്",
        "acc_imp_title": "പ്രധാന ഘടകങ്ങൾ",
    },
    "te": {
        "title": "భారత్ కార్డ్ షీల్డ్",
        "tagline": "కిరాణా దుకాణాలు, మాల్స్ మరియు అర్ధరాత్రి విదేశీ లావాదేవీలను తనిఖీ చేస్తున్నప్పుడు కుడివైపున ఉన్న రూపాయి డెబిట్ కార్డును గమనించండి.",
        "pill_text": "భారతీయ రూపాయి · ప్రత్యక్ష డెమో · డెమో బ్యాంక్ మాత్రమే",
        "how_to_read": "<b>వివరణ:</b> ప్రతి లావాదేవీకి 0% నుండి 100% రిస్క్ స్కోరు లభిస్తుంది. <b>50% లేదా అంతకంటే ఎక్కువ</b> ఉంటే OTP కోసం నిలిపివేయబడుతుంది.",
        "genuine": "నిజమైన లావాదేవీ",
        "fraud": "మోసపూరితంగా గుర్తించబడింది",
        
        "kpi_reviewed": "సమీక్షించిన లావాదేవీలు",
        "kpi_flagged": "మోసపూరిత లావాదేవీలు",
        "kpi_share": "మోసం శాతం",
        "kpi_typical": "సగటు మోసం మొత్తం",
        "sub_rows": "నమూనా ఖాతాలోని లావాదేవీలు",
        "sub_unauth": "అనధికారిక కార్డ్ ఖర్చులు",
        "sub_rare": "మోసం చాలా అరుదు",
        "sub_avg": "గుర్తించిన లావాదేవీల సగటు విలువ",
        "threat_title": "భద్రతా వ్యవస్థ మరియు మార్గదర్శకాలు",
        "threat_desc": "ఆర్థిక సైబర్ దాడులు అర్ధరాత్రి క్లియరింగ్ సమయాలను లక్ష్యంగా చేసుకుంటాయి.",
        "theory_1_title": "1. రాత్రివేళ దాడుల తీవ్రత",
        "theory_1_desc": "రాత్రి 11:00 నుండి ఉదయం 5:00 గంటల మధ్య అనధికారిక లావాదేవీలు జరుగుతాయి.",
        "theory_2_title": "2. డేటా అసమతుల్యత మరియు PCA స్కోరింగ్",
        "theory_2_desc": "మోడల్ రీకాల్ మరియు అసహజ ప్రవర్తన గుర్తింపుపై ఆధారపడుతుంది.",
        "dir_title": "జాతీయ భద్రతా పోర్టల్స్",
        "chart_time_title": "మోసాలు ఎప్పుడు ఎక్కువగా జరుగుతాయి?",
        "chart_time_sub": "అర్ధరాత్రి నుండి రాత్రి 11 గంటల వరకు.",
        "chart_pie_title": "సాధారణ vs మోసపూరిత లావాదేవీలు",
        "chart_pie_sub": "చాలావరకు లావాదేవీలు నిజమైనవే.",

        "hiw_kicker": "మార్గదర్శి",
        "hiw_title": "కంప్యూటర్లు మోసాలను ఎలా గుర్తిస్తాయి?",
        "hiw_sub": "నిర్దిష్ట పరిమితుల కంటే ప్రవర్తనా తీరు ఆధారంగా నిర్ణయం తీసుకుంటుంది.",
        "hiw_p1_title": "1. సాధారణ నియమాలు ఎందుకు విఫలమవుతాయి",
        "hiw_p1_body": "స్థిర పరిమితులు ఉన్న నియమాలను సులభంగా దాటవేయవచ్చు.",
        "hiw_p2_title": "2. మోసం చాలా అరుదు",
        "hiw_p2_body": "0.2% కంటే తక్కువ ఉండడం వల్ల సాధారణ ఖచ్చితత్వం సరిపోదు.",
        "hiw_p3_title": "3. తప్పిపోయిన మోసానికి ఎక్కువ జరిమానా",
        "hiw_p3_body": "మోసాన్ని గుర్తించడంలో వైఫల్యానికి 50 రెట్లు ఎక్కువ ప్రాధాన్యత.",
        "hiw_p4_title": "4. రీకాల్ ప్రాధాన్యత",
        "hiw_p4_body": "దొంగతనాన్ని అడ్డుకోవడానికి రీకాల్ (Recall) ముఖ్యం.",

        "sp_kicker": "చార్టులు",
        "sp_title": "సాధారణ మరియు మోసపూరిత ఖర్చులను పోల్చండి",
        "sp_sub": "వివరాలను ఎంచుకుని తేడాలను గమనించండి.",
        "sp_compare_label": "మీరు దేనిని పోల్చాలనుకుంటున్నారు?",
        "sp_bars_label": "బార్ల సంఖ్య",
        "sp_box_title": "ఖర్చు మొత్తం (లాగ్ స్కేల్)",
        "sp_box_caption": "చిన్న మరియు పెద్ద విలువలను ఒకే చోట చూపిస్తుంది.",
        "sp_corr_title": "పరస్పర సంబంధం",
        "sp_corr_caption": "ముదురు నీలం విలువలు కలిసి మారుతున్నాయని సూచిస్తుంది.",

        "chk_kicker": "ప్రత్యక్ష తనిఖీ",
        "chk_title": "చెల్లింపును తనిఖీ చేయండి",
        "chk_sub": "వివరాలను నమోదు చేసి రిస్క్ స్కోరును చూడండి.",
        "chk_stories": "ఉదాహరణలు",
        "btn_kirana": "కిరాణా షాపు · ₹3,735",
        "btn_atm": "రాత్రి 3 గంటల ఏటీఎం · ₹51,460",
        "btn_mall": "షాపింగ్ మాల్ · ₹1,03,750",
        "btn_reset": "రీసెట్",
        "chk_details": "చెల్లింపు వివరాలు",
        "lbl_amount": "మొత్తం (₹)",
        "lbl_hour": "సమయం",
        "lbl_loc": "ఎక్కడ చెల్లించారు?",
        "opt_india": "భారతదేశంలో",
        "opt_intl": "విదేశాలలో",
        "lbl_trust": "పరికర విశ్వసనీయత",
        "btn_run": "తనిఖీ చేయండి",
        "chk_result": "ఫలితం",
        "msg_prompt": "వివరాలను నమోదు చేసి తనిఖీ బటన్ నొక్కండి.",

        "about_kicker": "ప్రాజెక్ట్",
        "about_title": "ఈ డెమో ఏమిటి?",
        "about_sub": "కార్డ్ భద్రతా మెషిన్ లెర్నింగ్ ప్రాజెక్ట్.",
        "about_p1_title": "ఒక్క మాటలో",
        "about_p1_body": "30,000 లావాదేవీలను స్కోర్ చేసే ఏఐ మోడల్.",
        "about_p2_title": "నిర్మాణం",
        "about_p2_body": "120 చెట్లతో కూడిన రాండమ్ ఫారెస్ట్ మోడల్.",

        "acc_kicker": "స్కోర్‌బోర్డ్",
        "acc_title": "ఖచ్చితత్వం ఎంత?",
        "acc_sub": "పరీక్ష డేటా ఆధారంగా ఫలితాలు.",
        "acc_recall": "గుర్తించిన మోసాలు (Recall)",
        "acc_prec": "ఖచ్చితత్వం (Precision)",
        "acc_cm_title": "కన్ఫ్యూజన్ మాట్రిక్స్",
        "acc_imp_title": "కీలక లక్షణాలు",
    }
}

NAV_PAGES = [
    "Dashboard",
    "How it works",
    "Spending patterns",
    "Check a payment",
    "Currency converter",
    "Transaction history",
    "Ask the assistant",
    "About this system",
    "How accurate is it?",
    "Year-by-Year Fraud Timeline",
    "Indian Bank sites"
]

FEATURE_LABELS = {
    "Amount": "Payment amount (₹)",
    "Hour": "Time of day (hour)",
    "Device_Trust_Score": "Device trust score",
    "V1": "Behavior pattern 1 — spending pace",
    "V2": "Behavior pattern 2 — location change",
    "V3": "Behavior pattern 3 — device rhythm",
    "V4": "Behavior pattern 4 — app activity",
    "Is_International": "Paid from outside India?",
}

SCAN_DEFAULTS = {
    "amt": 75.0 * FX_RATE,
    "hour": 14,
    "intl": 0,
    "trust": 0.85,
    "v1": 0.1,
    "v2": -0.1,
    "v3": 0.0,
    "v4": 0.1,
}

SCAN_PRESETS = {
    "grocery": {
        "amt": 45.0 * FX_RATE, "hour": 16, "intl": 0, "trust": 0.95,
        "v1": 0.1, "v2": -0.1, "v3": 0.0, "v4": 0.1,
    },
    "skim": {
        "amt": 620.0 * FX_RATE, "hour": 3, "intl": 1, "trust": 0.08,
        "v1": -3.8, "v2": 3.2, "v3": -4.6, "v4": 3.9,
    },
    "retail": {
        "amt": 1250.0 * FX_RATE, "hour": 14, "intl": 0, "trust": 0.88,
        "v1": 0.3, "v2": 0.1, "v3": -0.2, "v4": 0.2,
    },
}

def format_inr(amount, paise=False):
    n = float(amount)
    if paise:
        s = f"{abs(n):.2f}"
        whole, frac = s.split(".")
        suffix = f".{frac}"
    else:
        whole = str(int(round(abs(n))))
        suffix = ""
    sign = "-" if n < 0 else ""
    if len(whole) <= 3:
        body = whole
    else:
        last3 = whole[-3:]
        rest = whole[:-3]
        chunks = []
        while rest:
            chunks.append(rest[-2:])
            rest = rest[:-2]
        body = ",".join(reversed(chunks)) + "," + last3
    return f"{sign}₹{body}{suffix}"

def to_model_amount(inr_value):
    return float(inr_value) / FX_RATE

def format_fx(amount, code=""):
    n = float(amount)
    code = (code or "").upper()
    if code in {"JPY", "KRW", "VND", "IDR", "IRR", "LAK", "PYG", "UGX", "RWF", "BIF", "GNF", "KHR", "MGA", "UZS"}:
        return f"{n:,.0f}"
    if code in {"KWD", "BHD", "OMR", "JOD"}:
        return f"{n:,.3f}"
    if abs(n) >= 1000:
        return f"{n:,.2f}"
    if abs(n) >= 1:
        return f"{n:,.4f}"
    return f"{n:.6f}".rstrip("0").rstrip(".")

@st.cache_data(ttl=3600)
def load_fx_rates():
    try:
        res = _fetch_inr_rates()
        if isinstance(res, tuple) and len(res) == 2:
            return res[0], res[1]
        elif isinstance(res, dict):
            return res, "Offline Reference"
    except Exception:
        pass
    return FALLBACK_INR_PER_UNIT, "Default Offline Reference"

def inject_styles():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #071628;
    --navy-2: #0C2C4A;
    --gold: #D4AF37;
    --gold-2: #F3E2A8;
    --bg: #EAF0F6;
    --card: #FFFFFF;
    --line: #D7E0EA;
    --text: #0F172A;
    --muted: #5B6B7C;
}

html, body, [class*="css"] { font-family: 'Outfit', sans-serif !important; }

.stApp {
    background:
        radial-gradient(900px 380px at 8% -8%, rgba(212,175,55,0.18), transparent 45%),
        radial-gradient(700px 420px at 92% 8%, rgba(7,22,40,0.14), transparent 48%),
        linear-gradient(180deg, #F7F4EC 0%, #EAF0F6 38%, #E4EBF3 100%) !important;
}

#MainMenu, header, footer, .stDeployButton { visibility: hidden; height: 0; }
[data-testid="stToolbar"] { display: none !important; }
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 3rem !important;
    max-width: 1440px;
    padding-right: 340px !important;
}

section[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, #07101C 0%, #0A2540 48%, #123A63 100%) !important;
    border-right: 1px solid rgba(212,175,55,0.38) !important;
}
section[data-testid="stSidebar"] * { color: #F8FAFC !important; }
section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 10px 12px !important;
    margin-bottom: 6px;
    font-weight: 500;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(212,175,55,0.14);
    border-color: rgba(212,175,55,0.45);
}
section[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] {
    background: linear-gradient(90deg, rgba(212,175,55,0.28), rgba(212,175,55,0.08)) !important;
    border-color: #D4AF37 !important;
}

.stButton > button {
    background: linear-gradient(180deg, #14385C 0%, #071628 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid #D4AF37 !important;
    border-radius: 10px !important;
    font-weight: 650 !important;
    letter-spacing: 0.01em;
    box-shadow: 0 8px 18px rgba(7,22,40,0.18);
}
.stButton > button:hover {
    background: linear-gradient(180deg, #F1D77A, #D4AF37) !important;
    color: #071628 !important;
}

.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border-radius: 10px !important;
}

div[data-testid="stChatMessage"] {
    background: #FFFFFF;
    border: 1px solid #D7E0EA;
    border-radius: 14px;
}

.brand-lockup {
    display: flex; align-items: center; gap: 12px;
    padding: 4px 2px 16px 2px;
    border-bottom: 1px solid rgba(212,175,55,0.28);
    margin-bottom: 14px;
}
.brand-mark {
    width: 40px; height: 40px; border-radius: 12px;
    background: linear-gradient(135deg, #F3E2A8, #D4AF37 55%, #8A6A12);
    color: #071628; font-weight: 800; font-size: 15px;
    display: flex; align-items: center; justify-content: center;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.5);
}
.top-banner {
    position: relative;
    overflow: hidden;
    background:
        linear-gradient(115deg, #07101C 0%, #0A2540 42%, #1A4A78 78%, #3A2A12 100%);
    color: #fff; padding: 26px 28px; border-radius: 18px;
    margin-bottom: 16px;
    box-shadow: 0 18px 40px rgba(7,22,40,0.22);
    border: 1px solid rgba(212,175,55,0.42);
}
.top-banner:after {
    content: "";
    position: absolute; right: -40px; top: -50px;
    width: 220px; height: 220px; border-radius: 50%;
    background: radial-gradient(circle, rgba(212,175,55,0.28), transparent 68%);
}
.top-banner h1 {
    font-family: 'Cormorant Garamond', serif;
    font-size: 34px; margin: 0 0 4px 0; letter-spacing: 0.3px;
}
.top-banner p { margin: 0; color: #D6E4F5; font-size: 14px; max-width: 640px; }
.pill {
    display: inline-block; font-size: 11px; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
    background: rgba(212,175,55,0.16); color: #F3E2A8;
    border: 1px solid rgba(212,175,55,0.45);
    padding: 5px 11px; border-radius: 999px; margin-bottom: 8px;
}
.note {
    background: linear-gradient(90deg, #FFF9E8, #FFFDF5);
    border: 1px solid #EBD9A0; color: #6B4E12;
    padding: 13px 16px; border-radius: 12px; font-size: 13.5px; margin-bottom: 18px;
}
.kpi {
    background: linear-gradient(180deg, #FFFFFF, #F8FBFE);
    border: 1px solid var(--line);
    border-radius: 16px; padding: 18px;
    box-shadow: 0 10px 24px rgba(15,23,42,0.05);
    position: relative; overflow: hidden; height: 100%;
}
.kpi:before {
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
    background: linear-gradient(180deg, #D4AF37, #0A2540);
}
.kpi .k { font-size: 11px; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }
.kpi .v { font-size: 26px; font-weight: 800; color: var(--navy); line-height: 1.15; }
.kpi .d { font-size: 12.5px; color: var(--muted); margin-top: 6px; line-height: 1.4; }
.panel {
    background: linear-gradient(180deg, #FFFFFF, #F9FBFD);
    border: 1px solid var(--line);
    border-radius: 16px; padding: 18px 20px;
    box-shadow: 0 8px 20px rgba(15,23,42,0.04);
    margin-bottom: 16px;
}
.panel h3 { margin: 0 0 6px 0; font-size: 16px; color: var(--navy); }
.panel p { color: #334155; font-size: 14.5px; line-height: 1.7; }
.page-kicker { font-size: 12px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #B0891A; margin-bottom: 4px; }
.page-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 34px; font-weight: 700; color: var(--navy); margin: 0 0 6px 0;
}
.page-sub { color: var(--muted); font-size: 15px; margin: 0 0 18px 0; max-width: 760px; }
.login-wrap { max-width: 520px; margin: 6vh auto 0 auto; padding-right: 0 !important; }
.verdict-bad, .verdict-ok { padding: 14px 16px; border-radius: 12px; margin-top: 8px; }
.verdict-bad { background: #FEF2F2; border: 1px solid #FECACA; border-left: 5px solid #DC2626; }
.verdict-ok { background: #F0FDF4; border: 1px solid #BBF7D0; border-left: 5px solid #16A34A; }

.plastic-wrap {
    position: fixed;
    right: 22px;
    top: 92px;
    width: 292px;
    z-index: 40;
}
.plastic {
    position: relative;
    width: 292px;
    height: 184px;
    border-radius: 18px;
    color: #F8F1D8;
    background:
        linear-gradient(125deg, #0B1C33 0%, #163A63 28%, #C9A227 58%, #3A2710 86%, #08111D 100%);
    box-shadow:
        0 24px 40px rgba(7, 16, 28, 0.38),
        inset 0 1px 0 rgba(255,255,255,0.35),
        inset 0 -18px 28px rgba(0,0,0,0.28);
    overflow: hidden;
    transform: perspective(900px) rotateY(-12deg) rotateX(8deg);
    transition: transform 0.4s ease;
}
.plastic:hover { transform: perspective(900px) rotateY(-2deg) rotateX(2deg) scale(1.03); }
.plastic .glint {
    position: absolute; inset: -40%;
    background: linear-gradient(115deg, transparent 35%, rgba(255,255,255,0.28) 50%, transparent 62%);
    animation: sheen 4.8s linear infinite;
}
@keyframes sheen { 0% { transform: translateX(-30%); } 100% { transform: translateX(30%); } }
.plastic .tri {
    position: absolute; top: 0; left: 0; right: 0; height: 5px;
    background: linear-gradient(90deg, #FF9933 0 33%, #FFFFFF 33% 66%, #138808 66% 100%);
    opacity: 0.9;
}
.plastic .row { display: flex; justify-content: space-between; align-items: center; padding: 14px 16px 0 16px; }
.plastic .bank {
    font-family: 'Cormorant Garamond', serif;
    font-weight: 700; font-size: 18px; letter-spacing: 0.6px;
}
.plastic .net {
    font-size: 11px; font-weight: 800; letter-spacing: 0.16em;
    border: 1px solid rgba(243,226,168,0.55);
    padding: 3px 7px; border-radius: 999px; color: #F3E2A8;
}
.chip {
    width: 42px; height: 32px; margin: 18px 0 10px 16px; border-radius: 6px;
    background: linear-gradient(135deg, #E6C567, #B8860B 40%, #F8E7A8 70%, #8A6A12);
    box-shadow: inset 0 0 0 1px rgba(80,50,0,0.35);
    position: relative;
}
.chip:before, .chip:after {
    content: ""; position: absolute; left: 6px; right: 6px; height: 1px; background: rgba(80,50,0,0.35);
}
.chip:before { top: 10px; }
.chip:after { top: 21px; }
.contactless {
    position: absolute; left: 66px; top: 58px; width: 22px; height: 22px;
    border-right: 2px solid rgba(243,226,168,0.8);
    border-radius: 50%;
    box-shadow: 6px 0 0 -4px rgba(243,226,168,0.7), 12px 0 0 -7px rgba(243,226,168,0.55);
}
.pan {
    font-family: ui-monospace, Consolas, monospace;
    letter-spacing: 0.18em; font-size: 15px; padding: 0 16px;
    text-shadow: 0 1px 0 rgba(0,0,0,0.35);
}
.meta { display: flex; justify-content: space-between; padding: 12px 16px 0 16px; font-size: 10px; letter-spacing: 0.08em; }
.meta b { display: block; font-size: 12px; letter-spacing: 0.12em; margin-top: 2px; color: #FFF8DC; }
.card-caption {
    margin-top: 14px; color: #334155; font-size: 12.5px; line-height: 1.5;
    background: rgba(255,255,255,0.78);
    border: 1px solid #E2E8F0; border-radius: 14px; padding: 12px 14px;
    backdrop-filter: blur(8px);
}
.card-caption strong { color: #071628; }

@media (max-width: 1200px) {
    .block-container { padding-right: 2rem !important; }
    .plastic-wrap { position: static; width: 100%; margin: 0 0 18px 0; }
    .plastic { transform: none; margin: 0 auto; }
}

@keyframes slideFromRightToLeft {
    0% { opacity: 0; transform: translateX(90px) scale(0.96); }
    60% { opacity: 0.8; transform: translateX(-12px) scale(1.01); }
    100% { opacity: 1; transform: translateX(0) scale(1); }
}

.card-transition-wrapper {
    animation: slideFromRightToLeft 0.65s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    will-change: transform, opacity;
}
</style>
        """,
        unsafe_allow_html=True,
    )

def apply_plot(fig, height=None):
    fig.update_layout(**PLOT_LAYOUT)
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor="#EEF2F7", zeroline=False)
    fig.update_yaxes(gridcolor="#EEF2F7", zeroline=False)
    return fig

def page_intro(kicker, title, subtitle):
    st.markdown(
        f"""
        <div class="page-kicker">{kicker}</div>
        <div class="page-title">{title}</div>
        <p class="page-sub">{subtitle}</p>
        """,
        unsafe_allow_html=True,
    )

def kpi_card(label, value, hint, accent=None):
    color = accent or "#071628"
    st.markdown(
        f"""
        <div class="kpi">
            <div class="k">{label}</div>
            <div class="v" style="color:{color};">{value}</div>
            <div class="d">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_credit_card(amount_inr=None, hour=None, intl=0, risk=None):
    if amount_inr is None:
        amount_inr = st.session_state.get("scan_amt", SCAN_DEFAULTS["amt"])
    if hour is None:
        hour = st.session_state.get("scan_hour", 14)
    loc = "INDIA DEBIT" if int(intl) == 0 else "INTL USE"
    amt = format_inr(amount_inr)
    hh = f"{int(hour):02d}:00"
    if risk is None:
        status = "READY"
        status_col = "#F3E2A8"
    elif risk >= 50:
        status = "HOLD"
        status_col = "#FCA5A5"
    else:
        status = "CLEAR"
        status_col = "#86EFAC"
    st.markdown(
        f"""
        <div class="plastic-wrap card-transition-wrapper">
            <div class="plastic">
                <div class="glint"></div>
                <div class="tri"></div>
                <div class="row">
                    <div class="bank">Bharat Card Shield</div>
                    <div class="net">{loc}</div>
                </div>
                <div class="chip"></div>
                <div class="contactless"></div>
                <div class="pan">5412&nbsp;&nbsp;••••&nbsp;&nbsp;••••&nbsp;&nbsp;4291</div>
                <div class="meta">
                    <div>CARDHOLDER<b> NULL</b></div>
                    <div>VALID THRU<b>08/29</b></div>
                    <div>TIME<b>{hh}</b></div>
                </div>
            </div>
            <div class="card-caption">
                <strong>{amt}</strong> will be charged on this demo card<br>
                Network: BharatPay Platinum &nbsp;·&nbsp; Status: <span style="color:{status_col};font-weight:800;">{status}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def apply_scan_values(values):
    st.session_state["scan_amt"] = float(values["amt"])
    st.session_state["scan_hour"] = int(values["hour"])
    st.session_state["scan_intl"] = int(values["intl"])
    st.session_state["scan_trust"] = float(values["trust"])
    st.session_state["scan_v1"] = float(values["v1"])
    st.session_state["scan_v2"] = float(values["v2"])
    st.session_state["scan_v3"] = float(values["v3"])
    st.session_state["scan_v4"] = float(values["v4"])
    st.session_state["scan_result"] = None

@st.cache_resource
def load_model_assets():
    model_path = "models/fraud_model.pkl"
    metrics_path = "models/model_metrics.pkl"
    if os.path.exists(model_path) and os.path.exists(metrics_path):
        return joblib.load(model_path), joblib.load(metrics_path)
    return None, None

@st.cache_data
def load_transaction_data():
    data_path = "data/transactions.csv"
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

inject_styles()
model, metrics = load_model_assets()
df_transactions = load_transaction_data()

if model is None or df_transactions is None or metrics is None:
    st.error("Please run `python train.py` first so it can create model assets and dataset.")
    st.stop()

df_transactions = df_transactions.copy()
df_transactions["Amount_INR"] = df_transactions["Amount"] * FX_RATE

expected_features = metrics.get(
    "features", [c for c in df_transactions.columns if c not in ("Class", "Amount_INR")]
)

# Check if user arrived verified from login.html
query_params = st.query_params
if query_params.get("auth") == "verified":
    st.session_state["authenticated"] = True

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "scan_amt" not in st.session_state or float(st.session_state.get("scan_amt", 0)) < 200:
    apply_scan_values(SCAN_DEFAULTS)
if "scan_result" not in st.session_state:
    st.session_state["scan_result"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        {"role": "assistant", "content": "Namaste. I can explain this card-fraud check in everyday language."}
    ]

def render_login():
    st.markdown('<div class="login-wrap">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="top-banner" style="text-align:center;">
            <div class="pill">Staff gateway · India rupee demo</div>
            <h1>Bharat Card Shield</h1>
            <p>Sign in to review card payments in Indian rupees before they leave the account.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.form("auth_box"):
        emp_id = st.text_input("Staff ID", placeholder="admin")
        emp_pwd = st.text_input("Password", type="password", placeholder="Enter passcode")
        submitted = st.form_submit_button("Sign in", use_container_width=True)
        if submitted:
            if emp_id.strip() == "admin" and emp_pwd == "fraud2026":
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Demo login: admin / fraud2026")
    st.markdown("</div>", unsafe_allow_html=True)

if st.query_params.get("auth") == "verified":
    st.session_state["authenticated"] = True

if not st.session_state.get("authenticated", False):
    st.markdown(
        """
        <div style="text-align: center; padding: 60px 20px; font-family: sans-serif;">
            <h2 style="color: #0A2540; margin-bottom: 12px;">Authentication Required</h2>
            <p style="color: #4A5568; margin-bottom: 24px;">Please sign in through the official gateway to access Bharat Card Shield.</p>
            <a href="https://basilahm2006.github.io/bharat-card-shield/login.html" 
               target="_self"
               style="background-color: #F5A623; color: #000; padding: 12px 28px; border-radius: 8px; text-decoration: none; font-weight: bold; display: inline-block;">
                Go to Secure Login Portal →
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()
st.sidebar.markdown(
    """
    <div class="brand-lockup">
        <div class="brand-mark">₹</div>
        <div>
            <div style="font-weight:800;font-size:15px;color:#fff;">Bharat Card Shield</div>
            <div style="font-size:12px;color:#CBD5E1;">National Card Defense</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------- LOCALIZED SIDEBAR -----------------
st.sidebar.markdown("### 🌐 Language / மொழி")
selected_lang_label = st.sidebar.selectbox("Interface Language", list(LANGUAGES.keys()), index=0)
lang = LANGUAGES[selected_lang_label]
txt = I18N[lang]

NAV_LABELS = {
    "en": NAV_PAGES,
    "ta": ["டாஷ்போர்டு", "இது எவ்வாறு செயல்படுகிறது", "செலவு முறைகள்", "பரிவர்த்தனை சரிபார்ப்பு", "நாணய மாற்றி", "பரிவர்த்தனை வரலாறு", "உதவியாளரிடம் கேளுங்கள்", "அமைப்பு பற்றி", "துல்லியம்", "வருடாந்திர மோசடி காலவரிசை", "இந்திய வங்கி தளங்கள்"],
    "hi": ["डैशबोर्ड", "यह कैसे काम करता है", "खर्च पैटर्न", "भुगतान जांचें", "मुद्रा परिवर्तक", "लेन-देन इतिहास", "सहायक से पूछें", "प्रणाली के बारे में", "सटीकता", "वार्षिक धोखाधड़ी समयरेखा", "भारतीय बैंक साइटें"],
    "ur": ["ڈیش بورڈ", "یہ کیسے کام کرتا ہے", "خرچ کے پیٹرن", "ادائیگی چیک کریں", "کرنسی کنورٹر", "لین دین کی تاریخ", "معاون سے پوچھیں", "سسٹم کے بارے میں", "درستگی", "سال بہ سال فراڈ ٹائم لائن", "ہندوستانی بینک سائٹس"],
    "ml": ["ഡാഷ്‌ബോർഡ്", "ഇത് എങ്ങനെ പ്രവർത്തിക്കുന്നു", "ചെലവ് രീതികൾ", "പേയ്‌മെന്റ് പരിശോധിക്കുക", "കറൻസി കൺവെർട്ടർ", "ഇടപാട് ചരിത്രം", "സഹായിയോട് ചോദിക്കുക", "സിസ്റ്റത്തെക്കുറിച്ച്", "കൃത്യത", "തട്ടിപ്പ് ടൈംലൈൻ", "ഇന്ത്യൻ ബാങ്ക് സൈറ്റുകൾ"],
    "te": ["డాష్‌బోర్డ్", "ఇది ఎలా పనిచేస్తుంది", "ఖర్చు విధానాలు", "చెల్లింపు తనిఖీ", "కరెన్సీ కన్వర్టర్", "లావాదేవీ చరిత్ర", "సహాయకుడిని అడగండి", "సిస్టమ్ గురించి", "ఖచ్చితత్వం", "మోసాల కాలక్రమం", "భారతీయ బ్యాంక్ సైట్లు"]
}

current_labels = NAV_LABELS.get(lang, NAV_PAGES)
label_to_choice = dict(zip(current_labels, NAV_PAGES))
selected_nav_label = st.sidebar.radio("Navigation", current_labels, index=0)
menu_choice = label_to_choice[selected_nav_label]

if st.sidebar.button("Sign out", use_container_width=True):
    st.session_state["authenticated"] = False
    st.rerun()

# ----------------- DYNAMIC BANNER & NOTICE -----------------
st.markdown(
    f"""
    <div class="top-banner">
        <div style="display:flex;justify-content:space-between;gap:16px;align-items:center;flex-wrap:wrap;position:relative;z-index:1;">
            <div>
                <div class="pill">{txt['pill_text']}</div>
                <h1>{txt['title']}</h1>
                <p>{txt['tagline']}</p>
            </div>
            <div style="text-align:right;font-size:12.5px;color:#D6E4F5;line-height:1.7;">
                <div>Network status: <b style="color:#86EFAC;">Online</b></div>
                <div>Currency: <b>INR (₹)</b></div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(f'<div class="note">{txt["how_to_read"]}</div>', unsafe_allow_html=True)

render_credit_card(
    amount_inr=st.session_state.get("scan_amt"),
    hour=st.session_state.get("scan_hour"),
    intl=st.session_state.get("scan_intl", 0),
    risk=st.session_state.get("scan_result"),
)

# ==========================================
# 1. DASHBOARD
# ==========================================
if menu_choice == "Dashboard":
    page_intro("Overview", txt['title'], txt['tagline'])
    total_records = len(df_transactions)
    fraud_records = int(df_transactions["Class"].sum())
    fraud_rate = (fraud_records / total_records) * 100 if total_records > 0 else 0
    avg_fraud_amt = df_transactions[df_transactions["Class"] == 1]["Amount_INR"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card(txt["kpi_reviewed"], f"{total_records:,}", txt["sub_rows"])
    with c2:
        kpi_card(txt["kpi_flagged"], f"{fraud_records:,}", txt["sub_unauth"], accent="#DC2626")
    with c3:
        kpi_card(txt["kpi_share"], f"{fraud_rate:.2f}%", txt["sub_rare"])
    with c4:
        kpi_card(txt["kpi_typical"], format_inr(avg_fraud_amt), txt["sub_avg"])

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    col_intel, col_charts = st.columns([1.1, 0.9])

    with col_intel:
        st.markdown(f"""
        <div style="font-size: 16px; font-weight: 700; color: #0A2540; margin-bottom: 6px;">{txt['threat_title']}</div>
        <div style="font-size: 13px; color: #475569; line-height: 1.5; margin-bottom: 14px;">{txt['threat_desc']}</div>
        
        <!-- Theory 1 -->
        <div style="background: #F8FAFC; border-left: 3px solid #0A2540; padding: 12px 14px; border-radius: 4px; margin-bottom: 10px;">
            <div style="font-size: 12px; font-weight: 700; color: #0A2540;">{txt['theory_1_title']}</div>
            <div style="font-size: 12px; color: #334155; margin-top: 2px;">{txt['theory_1_desc']}</div>
        </div>
        
        <!-- Theory 2 -->
        <div style="background: #F8FAFC; border-left: 3px solid #C59B27; padding: 12px 14px; border-radius: 4px; margin-bottom: 16px;">
            <div style="font-size: 12px; font-weight: 700; color: #854D0E;">{txt['theory_2_title']}</div>
            <div style="font-size: 12px; color: #334155; margin-top: 2px;">{txt['theory_2_desc']}</div>
        </div>
        """, unsafe_allow_html=True)

        # -------------------------------------------------------------
        # PAST FRAUD INVESTIGATION DOSSIERS & NEWS REPORTS
        # -------------------------------------------------------------
        st.markdown("<div style='font-size: 13px; font-weight: 700; color: #0A2540; margin-bottom: 8px;'>RECENT FRAUD CASE REPORTS & NEWS COVERAGE</div>", unsafe_allow_html=True)

        fraud_news_links = [
            {
                "title": "₹94 Cr Cosmos Bank ATM Switch Malware Attack",
                "source": "The Economic Times Investigation",
                "tag": "ATM Proxy Heist",
                "url": "https://economictimes.indiatimes.com/industry/banking/finance/banking/the-anatomy-of-cosmos-bank-heist/articleshow/65487771.cms",
                "desc": "Attackers cloned cards across 28 countries, triggering simultaneous withdrawals during off-peak hours.",
                "color": "#DC2626"
            },
            {
                "title": "₹70 Crore Multi-Tier Mule Account Network Busted",
                "source": "The Hindu Crime Bureau",
                "tag": "Mule Ring",
                "url": "https://www.thehindu.com/news/national/telangana/45-lakh-for-old-1-note-hyderabad-cyber-fraud-probe-uncovers-70-crore-mule-account-network/article71571816.ece",
                "desc": "Layering across hundreds of student and dormant accounts within seconds of transfer.",
                "color": "#0284C7"
            },
            {
                "title": "Debit Card ATM Cloning & Skimming Gang Convicted",
                "source": "The Hindu Legal Records",
                "tag": "Skimmer Gang",
                "url": "https://www.thehindu.com/news/cities/Coimbatore/coimbatore-court-convicts-man-who-headed-debit-card-cloning-gang/article69960608.ece",
                "desc": "Covert skimmers duplicated magnetic stripe data to clone 61 customer debit cards.",
                "color": "#854D0E"
            },
            {
                "title": "Delhi Police Busts ₹2.6 Cr Credit Card Fraud Scam",
                "source": "Telangana Today Cyber Desk",
                "tag": "Call Centre Breach",
                "url": "https://telanganatoday.com/delhi-police-bust-rs-2-6-crore-sbi-credit-card-scam-18-cyber-crooks-arrested",
                "desc": "Syndicate obtained customer card records through rogue insiders to orchestrate unauthorized charges.",
                "color": "#0A2540"
            }
        ]

        for item in fraud_news_links:
            st.markdown(f"""
            <a href="{item['url']}" target="_blank" style="text-decoration: none; display: block; margin-bottom: 8px;">
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 14px; display: flex; justify-content: space-between; align-items: center; transition: all 0.2s; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                    <div>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 9px; font-weight: 700; background: #F1F5F9; color: {item['color']}; padding: 1px 6px; border-radius: 3px;">{item['tag']}</span>
                            <span style="font-size: 13px; font-weight: 700; color: #0A2540;">{item['title']}</span>
                        </div>
                        <div style="font-size: 11px; color: #64748B; margin-top: 2px;">{item['desc']}</div>
                        <div style="font-size: 10px; color: #0284C7; font-weight: 600; margin-top: 3px;">Source: {item['source']}</div>
                    </div>
                    <span style="font-size: 14px; color: #0A2540; font-weight: 800; margin-left: 10px;">&nearr;</span>
                </div>
            </a>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 13px; font-weight: 700; color: #0A2540; margin-bottom: 8px;'>{txt['dir_title']}</div>", unsafe_allow_html=True)

        external_portals = [
            {
                "title": "National Cyber Crime Reporting Portal",
                "tag": "Helpline 1930 / I4C",
                "url": "https://cybercrime.gov.in/",
                "desc": "Ministry of Home Affairs portal for immediate reporting and interbank freeze requests.",
                "color": "#DC2626"
            },
            {
                "title": "RBI Kehta Hai – Financial Awareness",
                "tag": "Reserve Bank of India",
                "url": "https://rbikehtahai.rbi.org.in/",
                "desc": "Official regulatory directives on safe digital banking, unauthorized debit liability, and OTP safety.",
                "color": "#0284C7"
            },
            {
                "title": "CERT-In Financial Cyber Advisories",
                "tag": "Government Security Desk",
                "url": "https://www.cert-in.org.in/",
                "desc": "National security alerts on critical vulnerabilities in banking software and Android malware strains.",
                "color": "#0A2540"
            }
        ]

        for p in external_portals:
            st.markdown(f"""
            <a href="{p['url']}" target="_blank" style="text-decoration: none; display: block; margin-bottom: 8px;">
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 14px; display: flex; justify-content: space-between; align-items: center; transition: all 0.2s; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                    <div>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 9px; font-weight: 700; background: #F1F5F9; color: {p['color']}; padding: 1px 6px; border-radius: 3px;">{p['tag']}</span>
                            <span style="font-size: 13px; font-weight: 700; color: #0A2540;">{p['title']}</span>
                        </div>
                        <div style="font-size: 11px; color: #64748B; margin-top: 2px;">{p['desc']}</div>
                    </div>
                    <span style="font-size: 14px; color: #0A2540; font-weight: 800; margin-left: 10px;">&nearr;</span>
                </div>
            </a>
            """, unsafe_allow_html=True)
    with col_charts:
        st.markdown(f"##### {txt['chart_time_title']}")
        st.caption(txt['chart_time_sub'])
        hourly_data = df_transactions[df_transactions["Class"] == 1].groupby("Hour").size().reindex(range(24), fill_value=0).reset_index(name="Incidents")
        fig_time = px.bar(hourly_data, x="Hour", y="Incidents")
        fig_time.update_traces(marker_color="#0A2540")
        fig_time = apply_plot(fig_time, 240)
        st.plotly_chart(fig_time, use_container_width=True)

        st.markdown(f"##### {txt['chart_pie_title']}")
        st.caption(txt['chart_pie_sub'])
        class_counts = df_transactions["Class"].value_counts().rename(index={0: txt["genuine"], 1: txt["fraud"]})
        fig_donut = px.pie(values=class_counts.values, names=class_counts.index, hole=0.62, color=class_counts.index, color_discrete_map={txt["genuine"]: "#0A2540", txt["fraud"]: "#DC2626"})
        fig_donut = apply_plot(fig_donut, 240)
        st.plotly_chart(fig_donut, use_container_width=True)

# ==========================================
# 2. HOW IT WORKS
# ==========================================
elif menu_choice == "How it works":
    page_intro(txt['hiw_kicker'], txt['hiw_title'], txt['hiw_sub'])
    l, r = st.columns(2)
    with l:
        st.markdown(f'<div class="panel"><h3>{txt["hiw_p1_title"]}</h3><p>{txt["hiw_p1_body"]}</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="panel"><h3>{txt["hiw_p2_title"]}</h3><p>{txt["hiw_p2_body"]}</p></div>', unsafe_allow_html=True)
    with r:
        st.markdown(f'<div class="panel"><h3>{txt["hiw_p3_title"]}</h3><p>{txt["hiw_p3_body"]}</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="panel"><h3>{txt["hiw_p4_title"]}</h3><p>{txt["hiw_p4_body"]}</p></div>', unsafe_allow_html=True)

# ==========================================
# 3. SPENDING PATTERNS
# ==========================================
elif menu_choice == "Spending patterns":
    page_intro(txt["sp_kicker"], txt["sp_title"], txt["sp_sub"])
    plot_source = df_transactions.copy()
    numeric_columns = [c for c in plot_source.select_dtypes(include=[np.number]).columns if c not in ("Class", "Amount_INR")]
    c1, c2 = st.columns([2, 1])
    with c1:
        chosen_var = st.selectbox(txt["sp_compare_label"], numeric_columns, format_func=lambda x: FEATURE_LABELS.get(x, x))
    with c2:
        bin_count = st.slider(txt["sp_bars_label"], 10, 60, 25, 5)

    chart_df = plot_source.copy()
    chart_df["Status"] = chart_df["Class"].map({0: txt["genuine"], 1: txt["fraud"]})
    fig_hist = px.histogram(chart_df, x=chosen_var, color="Status", nbins=bin_count, barmode="overlay", color_discrete_map={txt["genuine"]: "#0A2540", txt["fraud"]: "#DC2626"})
    st.plotly_chart(apply_plot(fig_hist, 360), use_container_width=True)

# ==========================================
# 4. CHECK A PAYMENT
# ==========================================
elif menu_choice == "Check a payment":
    page_intro(txt["chk_kicker"], txt["chk_title"], txt["chk_sub"])
    st.markdown(f"##### {txt['chk_stories']}")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        if st.button(txt["btn_kirana"], use_container_width=True):
            apply_scan_values(SCAN_PRESETS["grocery"])
            st.rerun()
    with p2:
        if st.button(txt["btn_atm"], use_container_width=True):
            apply_scan_values(SCAN_PRESETS["skim"])
            st.rerun()
    with p3:
        if st.button(txt["btn_mall"], use_container_width=True):
            apply_scan_values(SCAN_PRESETS["retail"])
            st.rerun()
    with p4:
        if st.button(txt["btn_reset"], use_container_width=True):
            apply_scan_values(SCAN_DEFAULTS)
            st.rerun()

    f_left, f_right = st.columns([1.15, 1])
    with f_left:
        st.markdown(f"##### {txt['chk_details']}")
        st.number_input(txt["lbl_amount"], min_value=50.0, max_value=2000000.0, step=50.0, key="scan_amt")
        st.slider(txt["lbl_hour"], 0, 23, key="scan_hour")
        st.selectbox(txt["lbl_loc"], [0, 1], format_func=lambda x: txt["opt_india"] if x == 0 else txt["opt_intl"], key="scan_intl")
        st.slider(txt["lbl_trust"], 0.0, 1.0, step=0.05, key="scan_trust")
        run_inference = st.button(txt["btn_run"], use_container_width=True, type="primary")

    with f_right:
        st.markdown(f"##### {txt['chk_result']}")
        if run_inference:
            payload = pd.DataFrame([{
                "Hour": st.session_state["scan_hour"],
                "Amount": to_model_amount(st.session_state["scan_amt"]),
                "V1": st.session_state["scan_v1"],
                "V2": st.session_state["scan_v2"],
                "V3": st.session_state["scan_v3"],
                "V4": st.session_state["scan_v4"],
                "Is_International": st.session_state["scan_intl"],
                "Device_Trust_Score": st.session_state["scan_trust"],
            }])[expected_features]
            prob = float(model.predict_proba(payload)[0][1])
            st.session_state["scan_result"] = prob * 100

        risk_score = st.session_state["scan_result"]
        shown_amt = format_inr(st.session_state["scan_amt"])
        if risk_score is None:
            st.info(txt["msg_prompt"])
        else:
            dial_color = "#DC2626" if risk_score >= 50 else "#15803D"
            fig_dial = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                number={"suffix": "%", "font": {"color": dial_color, "size": 34}},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": dial_color}}
            ))
            fig_dial.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(fig_dial, use_container_width=True)

# ==========================================
# 5. CURRENCY CONVERTER
# ==========================================
elif menu_choice == "Currency converter":
    page_intro("Rupee desk", "Check any country’s money in Indian rupees", "Exchange rates conversion console.")
    rates, rate_stamp = load_fx_rates()
    st.caption(rate_stamp)
    countries = [(c, info["code"], info["name"]) for c, info in COUNTRY_CURRENCIES.items()]
    c_sel, c_dir = st.columns([2.2, 1])
    with c_sel:
        selected_item = st.selectbox("Country / currency", options=countries, format_func=lambda x: f"{x[0]} ({x[1]} - {x[2]})")
        country, code, ccy_name = selected_item[0], selected_item[1], selected_item[2]
    with c_dir:
        direction = st.radio("Convert", ["Foreign → Indian rupees", "Indian rupees → Foreign"])
        amount = st.number_input("Amount", min_value=0.0, value=1.0, step=1.0)
    inr_each = float(rates.get(code) or FALLBACK_INR_PER_UNIT.get(code) or 0)
    if inr_each > 0:
        val = amount * inr_each if direction.startswith("Foreign") else amount / inr_each
        st.markdown(f'<div class="panel"><h3>{country}</h3><h2>{val:,.2f}</h2></div>', unsafe_allow_html=True)

# ==========================================
# 6. TRANSACTION HISTORY
# ==========================================
elif menu_choice == "Transaction history":
    page_intro("Records", "Browse the sample rupee ledger", "Filter by status and amount in ₹.")
    st.dataframe(df_transactions.head(50), use_container_width=True)

# ==========================================
# 7. ASK THE ASSISTANT
# ==========================================
elif menu_choice == "Ask the assistant":
    page_intro("Help", "Ask a question about fraud checks", "Ask cybersecurity and fraud telemetry questions.")
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    user_prompt = st.chat_input("Ask a question…")
    if user_prompt:
        st.session_state["chat_history"].append({"role": "user", "content": user_prompt})
        st.session_state["chat_history"].append({"role": "assistant", "content": f"Analyzing: {user_prompt}"})
        st.rerun()

# ==========================================
# 8. ABOUT THIS SYSTEM
# ==========================================
elif menu_choice == "About this system":
    page_intro(txt["about_kicker"], txt["about_title"], txt["about_sub"])
    st.markdown(f'<div class="panel"><h3>{txt["about_p1_title"]}</h3><p>{txt["about_p1_body"]}</p></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="panel"><h3>{txt["about_p2_title"]}</h3><p>{txt["about_p2_body"]}</p></div>', unsafe_allow_html=True)

# ==========================================
# 9. HOW ACCURATE IS IT?
# ==========================================
elif menu_choice == "How accurate is it?":
    page_intro(txt["acc_kicker"], txt["acc_title"], txt["acc_sub"])
    b1, b2 = st.columns(2)
    with b1:
        kpi_card(txt["acc_recall"], f"{metrics.get('recall', 0)*100:.1f}%", "Recall")
    with b2:
        kpi_card(txt["acc_prec"], f"{metrics.get('precision', 0)*100:.1f}%", "Precision", accent="#15803D")

# ==============================================================
# 10. YEAR-BY-YEAR FRAUD TIMELINE
# ==============================================================
elif menu_choice == "Year-by-Year Fraud Timeline":
    st.markdown("""
    <div style="background: linear-gradient(90deg, #051A30 0%, #0A2540 60%, #153E6A 100%); color: #FFFFFF; padding: 18px 24px; border-radius: 8px; border-bottom: 3px solid #C59B27; margin-bottom: 16px;">
        <div style="font-size: 20px; font-weight: 700; color: #FFFFFF;">CHRONOLOGY OF FINANCIAL FRAUD & COUNTERMEASURES (2018 – 2026)</div>
        <div style="font-size: 13px; color: #C59B27; margin-top: 4px;">Evolution of Cybercrime Attack Vectors, Annual Impact (INR ₹), and Institutional Defenses</div>
    </div>
    """, unsafe_allow_html=True)

    timeline_records = [
        {
            "Year": 2018,
            "Era": "ATM Skimming & POS Clones",
            "Reported_Incidents_India": 26800,
            "Estimated_Loss_Cr_INR": 110,
            "Dominant_Vector": "Magnetic Stripe Cloning & Physical Skimmers",
            "How_Fraud_Took_Place": "Fraudsters attached covert skimmer hardware and pinhole cameras to physical ATM slots and POS swipe terminals to duplicate magnetic stripe data.",
            "Why_Old_Systems_Failed": "Static rule engines had no visibility over physical card integrity; any card with copied track-2 data was treated as authentic.",
            "How_It_Was_Handled": "Mandatory national migration to EMV dynamic chip-and-PIN cards, geofencing ATM withdrawals, and real-time terminal tamper alerts."
        },
        {
            "Year": 2020,
            "Era": "COVID-19 Phishing & SIM Swapping",
            "Reported_Incidents_India": 48200,
            "Estimated_Loss_Cr_INR": 235,
            "Dominant_Vector": "Telecom SIM Hijacking & Remote Screen Sharing",
            "How_Fraud_Took_Place": "Attackers tricked users into installing remote-access APKs (e.g., AnyDesk, TeamViewer) under the guise of pandemic KYC renewals or bribery of telecom staff for SIM duplication to intercept SMS OTPs.",
            "Why_Old_Systems_Failed": "Authentication relied entirely on SMS OTPs; once the phone number was hijacked, legacy banks had no secondary signal to challenge.",
            "How_It_Was_Handled": "Implementation of Device Binding (checking SIM IMSI vs device hardware ID) and blacklisting transactions when remote screen-mirroring apps are active."
        },
        {
            "Year": 2022,
            "Era": "UPI Social Engineering & QR Traps",
            "Reported_Incidents_India": 84500,
            "Estimated_Loss_Cr_INR": 410,
            "Dominant_Vector": "Collect-Request Spoofing & Fake Merchant QR Codes",
            "How_Fraud_Took_Place": "Victims selling secondhand goods online were sent 'Collect Requests' or reverse QR codes with deceptive messages claiming 'Scan to receive money'. Entering their UPI PIN immediately debited their accounts.",
            "Why_Old_Systems_Failed": "The user willingly authenticated the transaction with their correct PIN, bypassing cryptographic security.",
            "How_It_Was_Handled": "Redesigning banking UI to display bright red warnings on Collect Requests, mandatory 4-hour cooling periods for newly added Payees, and behavioral velocity limits."
        },
        {
            "Year": 2024,
            "Era": "Mule Accounts & Digital Arrest Scams",
            "Reported_Incidents_India": 132000,
            "Estimated_Loss_Cr_INR": 980,
            "Dominant_Vector": "Organized Money Mule Networks & Video Impersonation",
            "How_Fraud_Took_Place": "Syndicates recruited thousands of student/dormant accounts ('mules') to instantly layer and route stolen money across 8-10 tiers within seconds, accompanied by fake police video calls ('digital arrest').",
            "Why_Old_Systems_Failed": "Individual transactions appeared legitimate at single-bank levels; no single bank could see the multi-tier cross-bank rapid fund dispersal.",
            "How_It_Was_Handled": "Interbank rapid-freeze integration (I4C / 1930 portal integration), graph anomaly detection to map money mule clusters, and automated quarantine of rapid pass-through funds."
        },
        {
            "Year": 2026,
            "Era": "Generative AI Impersonation & Automated Bot Raids",
            "Reported_Incidents_India": 178000,
            "Estimated_Loss_Cr_INR": 1420,
            "Dominant_Vector": "AI Deepfake Voice Cloning & Distributed Micro-Bots",
            "How_Fraud_Took_Place": "Automated AI bots test compromised credentials across thousands of micro-merchants simultaneously, paired with synthetic biometric voice clones to authorize phone-banking transfers.",
            "Why_Old_Systems_Failed": "Traditional security checks cannot distinguish recorded voice clones or distributed micro-transactions under ₹500.",
            "How_It_Was_Handled": "National Financial Sentinel Architecture: Continuous PCA behavioral telemetry, weighted real-time random forest risk scoring, keystroke cadence analysis, and autonomous pre-settlement quarantine."
        }
    ]

    df_timeline = pd.DataFrame(timeline_records)

    col_t1, col_t2 = st.columns([1.6, 1])

    with col_t1:
        st.markdown("<div style='font-size: 13px; font-weight: 700; color: #0A2540; margin-bottom: 6px;'>REPORTED ANNUAL LOSS TRAJECTORY (₹ CRORES)</div>", unsafe_allow_html=True)
        fig_trend = px.line(
            df_timeline,
            x="Year",
            y="Estimated_Loss_Cr_INR",
            markers=True,
            text="Estimated_Loss_Cr_INR",
            template="simple_white"
        )
        fig_trend.update_traces(
            line_color="#DC2626",
            line_width=3,
            marker=dict(size=10, color="#0A2540", line=dict(width=2, color="#C59B27")),
            textposition="top center",
            texttemplate="₹%{text} Cr"
        )
        fig_trend.update_layout(
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis=dict(title="Year", tickmode="linear", tick0=2018, dtick=2),
            yaxis=dict(title="Estimated National Loss (₹ Crores)", gridcolor="#E2E8F0")
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_t2:
        st.markdown("<div style='font-size: 13px; font-weight: 700; color: #0A2540; margin-bottom: 6px;'>INCIDENT VOLUME GROWTH</div>", unsafe_allow_html=True)
        fig_vol = px.bar(
            df_timeline,
            x="Year",
            y="Reported_Incidents_India",
            text="Reported_Incidents_India",
            template="simple_white"
        )
        fig_vol.update_traces(marker_color="#0A2540", texttemplate="%{text:,}", textposition="outside")
        fig_vol.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title="Year", tickmode="linear", dtick=2),
            yaxis=dict(title="Annual Fraud Cases", gridcolor="#E2E8F0")
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown("---")
    st.markdown("<div style='font-size: 16px; font-weight: 700; color: #0A2540; margin-bottom: 10px;'>EXAMINE SPECIFIC ERA & DEFENSE STRATEGY</div>", unsafe_allow_html=True)

    selected_year = st.select_slider(
        "Select Year to Inspect Modus Operandi & Mitigation:",
        options=[2018, 2020, 2022, 2024, 2026],
        value=2026
    )

    year_data = next(item for item in timeline_records if item["Year"] == selected_year)

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-top: 3px solid #0A2540; padding: 16px; border-radius: 6px;">
            <div style="font-size: 11px; font-weight: 700; color: #64748B;">DOMINANT THREAT PROFILE</div>
            <div style="font-size: 16px; font-weight: 700; color: #0A2540; margin-top: 4px;">{year_data['Era']}</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-top: 3px solid #DC2626; padding: 16px; border-radius: 6px;">
            <div style="font-size: 11px; font-weight: 700; color: #64748B;">REPORTED CASES IN INDIA</div>
            <div style="font-size: 18px; font-weight: 700; color: #DC2626; margin-top: 4px;">{year_data['Reported_Incidents_India']:,} Incidents</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-top: 3px solid #C59B27; padding: 16px; border-radius: 6px;">
            <div style="font-size: 11px; font-weight: 700; color: #64748B;">ANNUAL LOSS ESTIMATE</div>
            <div style="font-size: 18px; font-weight: 700; color: #A16207; margin-top: 4px;">₹{year_data['Estimated_Loss_Cr_INR']} Crores</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c_how, c_sol = st.columns(2)
    with c_how:
        st.markdown(f"""
        <div style="background: #FFF5F5; border: 1px solid #FED7D7; border-left: 4px solid #DC2626; border-radius: 6px; padding: 20px;">
            <div style="font-size: 14px; font-weight: 700; color: #9B2C2C; margin-bottom: 8px;">HOW FRAUD TOOK PLACE IN {selected_year}</div>
            <div style="font-size: 13px; color: #4A5568; line-height: 1.6; margin-bottom: 12px;">
                <b>Modus Operandi:</b> {year_data['How_Fraud_Took_Place']}
            </div>
            <div style="font-size: 13px; color: #742A2A; line-height: 1.5; background: #FED7D7; padding: 10px; border-radius: 4px;">
                <b>Vulnerability:</b> {year_data['Why_Old_Systems_Failed']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_sol:
        st.markdown(f"""
        <div style="background: #F0FFF4; border: 1px solid #C6F6D5; border-left: 4px solid #38A169; border-radius: 6px; padding: 20px;">
            <div style="font-size: 14px; font-weight: 700; color: #22543D; margin-bottom: 8px;">HOW THE SYSTEM HANDLED & NEUTRALIZED IT</div>
            <div style="font-size: 13px; color: #2D3748; line-height: 1.6; margin-bottom: 12px;">
                <b>Strategic Countermeasures:</b> {year_data['How_It_Was_Handled']}
            </div>
            <div style="font-size: 13px; color: #22543D; line-height: 1.5; background: #C6F6D5; padding: 10px; border-radius: 4px;">
                <b>Operational Status:</b> Successfully integrated into automated pre-settlement threat rules.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 14px; font-weight: 700; color: #0A2540; margin-bottom: 8px;'>COMPLETE DECADE DEFENSE ROADMAP (2018 - 2026)</div>", unsafe_allow_html=True)

    summary_df = df_timeline[['Year', 'Era', 'Dominant_Vector', 'Estimated_Loss_Cr_INR']].copy()
    summary_df.columns = ['Year', 'Threat Era', 'Primary Vector', 'Est. Loss (₹ Crores)']
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ==============================================================
# 11. OFFICIAL INDIAN BANK PORTALS DIRECTORY
# ==============================================================
elif menu_choice == "Indian Bank sites":
    st.markdown("""
    <div style="background: linear-gradient(90deg, #051A30 0%, #0A2540 60%, #153E6A 100%); color: #FFFFFF; padding: 18px 24px; border-radius: 8px; border-bottom: 3px solid #C59B27; margin-bottom: 20px;">
        <div style="font-size: 20px; font-weight: 700; color: #FFFFFF;">VERIFIED NATIONAL BANKING DIRECTORY</div>
        <div style="font-size: 13px; color: #C59B27; margin-top: 4px;">Direct gateways to official RBI-regulated scheduled commercial banking sites (.bank.in)</div>
    </div>
    """, unsafe_allow_html=True)

    banks_data = [
        {
            "name": "State Bank of India (SBI)",
            "short": "SBI",
            "tag": "Public Sector / National Lead",
            "url": "https://sbi.bank.in/",
            "bg": "#1A5276",
            "accent": "#00AEEF",
            "logo_icon": "🏛️",
            "desc": "Largest public sector commercial bank in India. Personal & corporate net-banking portal."
        },
        {
            "name": "HDFC Bank",
            "short": "HDFC",
            "tag": "Private Sector / Commercial",
            "url": "https://www.hdfc.bank.in/",
            "bg": "#004C8F",
            "accent": "#ED1C24",
            "logo_icon": "🏢",
            "desc": "Premier private sector bank providing online corporate, retail banking, and card services."
        },
        {
            "name": "Axis Bank",
            "short": "AXIS",
            "tag": "Private Sector / Commercial",
            "url": "https://www.axis.bank.in/",
            "bg": "#97144D",
            "accent": "#F47920",
            "logo_icon": "💳",
            "desc": "Major private commercial bank providing digital accounts, retail loans, and secure UPI suites."
        },
        {
            "name": "Canara Bank",
            "short": "CANARA",
            "tag": "Public Sector Bank",
            "url": "https://www.canarabank.bank.in/",
            "bg": "#0091DA",
            "accent": "#FFD100",
            "logo_icon": "🪙",
            "desc": "Leading public sector institution with universal retail banking and digital payment desks."
        },
        {
            "name": "Indian Overseas Bank (IOB)",
            "short": "IOB",
            "tag": "Public Sector Bank",
            "url": "https://www.iob.bank.in/en/",
            "bg": "#004B87",
            "accent": "#C59B27",
            "logo_icon": "🌐",
            "desc": "Major state-owned financial institution specializing in retail, forex, and electronic remittances."
        }
    ]

    st.markdown("""
    <style>
    .bank-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: all 0.25s ease-in-out;
        text-decoration: none !important;
        box-shadow: 0 2px 6px rgba(10, 37, 64, 0.05);
    }
    .bank-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(10, 37, 64, 0.12);
        border-color: #C59B27;
    }
    .bank-avatar {
        width: 58px;
        height: 58px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        font-weight: 800;
        color: #FFFFFF;
        flex-shrink: 0;
        margin-right: 18px;
        border: 2px solid rgba(255, 255, 255, 0.9);
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }
    .bank-info {
        flex-grow: 1;
    }
    .bank-title {
        font-size: 17px;
        font-weight: 700;
        color: #0A2540;
        margin-bottom: 2px;
    }
    .bank-badge {
        display: inline-block;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748B;
        background: #F1F5F9;
        padding: 2px 8px;
        border-radius: 4px;
        margin-bottom: 6px;
    }
    .bank-desc {
        font-size: 12px;
        color: #475569;
        margin-top: 2px;
    }
    .bank-btn {
        background: #0A2540;
        color: #FFFFFF !important;
        padding: 9px 18px;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 600;
        white-space: nowrap;
        border: 1px solid #0A2540;
        transition: 0.2s;
        text-decoration: none !important;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .bank-card:hover .bank-btn {
        background: #C59B27;
        border-color: #C59B27;
        color: #051A30 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    for b in banks_data:
        st.markdown(f"""
        <a href="{b['url']}" target="_blank" class="bank-card">
            <div style="display: flex; align-items: center;">
                <div class="bank-avatar" style="background: linear-gradient(135deg, {b['bg']}, {b['accent']});">
                    {b['logo_icon']}
                </div>
                <div class="bank-info">
                    <span class="bank-badge">{b['tag']}</span>
                    <div class="bank-title">{b['name']}</div>
                    <div class="bank-desc">{b['desc']}</div>
                    <div style="font-size: 11px; color: #0284C7; margin-top: 4px; font-family: monospace;">🔗 {b['url']}</div>
                </div>
            </div>
            <div>
                <span class="bank-btn">Visit Official Site &rarr;</span>
            </div>
        </a>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top: 25px; padding: 14px 18px; background: #FEF3C7; border-left: 4px solid #D97706; border-radius: 6px; font-size: 12px; color: #78350F;">
        <b>Anti-Phishing Notice:</b> Always verify the top-level domain ends with <code>.bank.in</code> or <code>.co.in</code> to protect against typosquatting and impersonation attacks.
    </div>
    """, unsafe_allow_html=True)
