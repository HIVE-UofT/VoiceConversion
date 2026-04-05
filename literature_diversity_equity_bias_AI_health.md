# Literature Review: Diversity, Equity, and Bias in AI/ML for Health Research
## Relevance to Voice Conversion / Speech Processing for Health Applications

Compiled: 2026-03-31  
Focus: Papers addressing bias, fairness, equity, and diversity in health AI/ML, including voice/speech biomarker bias and ASR equity — directly relevant to voice biomarker research and speaker adaptation.

---

## Paper 1

**PMID:** 37380750  
**Title:** Algorithmic fairness in artificial intelligence for medicine and healthcare  
**Authors:** Richard J Chen, Judy J Wang, Drew F K Williamson, Tiffany Y Chen, Jana Lipkova, Ming Y Lu, Sharifa Sahai, Faisal Mahmood  
**Year:** 2023  
**Journal:** Nature Biomedical Engineering  
**DOI:** 10.1038/s41551-023-01056-8  
**Abstract:** The paper addresses how "insufficiently fair systems of artificial intelligence can undermine the delivery of equitable care." The authors examine algorithmic biases arising from data acquisition, genetic variation, and labeling inconsistencies in clinical settings. They document how "assessments of AI models stratified across subpopulations have revealed inequalities in how patients are diagnosed, treated and billed." The perspective explores mitigation strategies including disentanglement, federated learning, and explainability techniques for developing fair AI-based medical devices.

---

## Paper 2

**PMID:** 36084616  
**Title:** Algorithmic fairness in computational medicine  
**Authors:** Jie Xu, Yunyu Xiao, Wendy Hui Wang, Yue Ning, Elizabeth A Shenkman, Jiang Bian, Fei Wang  
**Year:** 2022  
**Journal:** EBioMedicine  
**DOI:** 10.1016/j.ebiom.2022.104250  
**Abstract:** This review examines how machine learning systems can introduce systematic biases when applied to clinical decision-making across different demographic groups. The authors note that "machine learning techniques may result in potential biases when making decisions for people in different subgroups, which can lead to detrimental effects on the health and well-being of specific demographic groups such as vulnerable ethnic minorities." The paper provides a comprehensive overview addressing three key areas: types of algorithmic bias, fairness measurement approaches, and strategies to reduce bias. The authors additionally summarize available software tools and libraries designed for bias assessment and mitigation.

---

## Paper 3

**PMID:** 37615031  
**Title:** Algorithmic fairness and bias mitigation for clinical machine learning with deep reinforcement learning  
**Authors:** Jenny Yang, Andrew A S Soltan, David W Eyre, David A Clifton  
**Year:** 2023  
**Journal:** Nature Machine Intelligence  
**DOI:** 10.1038/s42256-023-00697-3  
**Abstract:** The researchers developed a reinforcement learning framework to address biases in healthcare machine learning models. They evaluated their approach using COVID-19 prediction for emergency department patients, targeting hospital-specific and ethnicity-based prejudices in the dataset. Using a specialized reward function and training methodology, they demonstrated clinically viable screening performance while substantially enhancing outcome fairness compared to established benchmarks. The work included external validation across three independent hospitals and tested generalizability on intensive care discharge predictions.

---

## Paper 4

**PMID:** 36060496  
**Title:** Enabling Fairness in Healthcare Through Machine Learning  
**Authors:** Thomas Grote, Geoff Keeling  
**Year:** 2022  
**Journal:** Ethics and Information Technology  
**DOI:** 10.1007/s10676-022-09658-7  
**Abstract:** The authors examine how machine learning systems in healthcare decision-support might worsen health inequalities, while also noting that "algorithms trained on sufficiently diverse datasets could in principle combat health inequalities." They focus on a particular concern: when algorithmic performance for disadvantaged patient groups surpasses that for advantaged groups, creating apparent unfairness by standard ML metrics. The paper defends "affirmative algorithms" — systems trained on diverse data that perform better for traditionally disadvantaged populations. Their central argument: algorithmic fairness itself is not the key ethical concern; rather, what matters morally is "the fairness of final decisions, such as diagnoses, resulting from collaboration between clinicians and algorithms."

---

## Paper 5

**PMID:** 33981989  
**Title:** Addressing Fairness, Bias, and Appropriate Use of Artificial Intelligence and Machine Learning in Global Health  
**Authors:** Richard Ribón Fletcher, Audace Nakeshimana, Olusubomi Olubeko  
**Year:** 2021  
**Journal:** Frontiers in Artificial Intelligence  
**DOI:** 10.3389/frai.2020.561802  
**Abstract:** This editorial examines how AI and machine learning can responsibly serve global health, particularly in low- and middle-income countries. The authors establish three evaluation criteria: "APPROPRIATENESS is the process of deciding how the algorithm should be used in the local context, and properly matching the machine learning model to the target population." They address bias (systematic favoritism toward demographic groups) and fairness (examining impact across populations). The piece demonstrates these principles through a case study of ML applications for diagnosing pulmonary diseases in Pune, India, analyzing performance variations across gender and socioeconomic status.

---

## Paper 6

**PMID:** 38677633  
**Title:** A survey of recent methods for addressing AI fairness and bias in biomedicine  
**Authors:** Yifan Yang, Mingquan Lin, Han Zhao, Yifan Peng, Furong Huang, Zhiyong Lu  
**Year:** 2024  
**Journal:** Journal of Biomedical Informatics  
**DOI:** 10.1016/j.jbi.2024.104646  
**Abstract:** This review examines debiasing approaches in biomedical AI across natural language processing and computer vision domains. The authors conducted a systematic literature search covering January 2018–December 2023, ultimately reviewing 55 articles. Key findings indicate that bias originates from multiple sources including "insufficient data, sampling bias and the use of health-irrelevant features or race-adjusted algorithms." The paper categorizes existing remediation strategies into distributional methods (data augmentation, perturbation, reweighting, federated learning) and algorithmic methods (unsupervised representation learning, adversarial learning, disentangled representation learning, loss-based and causality-based methods). The authors emphasize that addressing bias during model development is essential for "accurate and reliable application of AI models in clinical settings."

---

## Paper 7

**PMID:** 35639450  
**Title:** Evaluation and Mitigation of Racial Bias in Clinical Machine Learning Models: Scoping Review  
**Authors:** Jonathan Huang, Galal Galal, Mozziyar Etemadi, Mahesh Vaidyanathan  
**Year:** 2022  
**Journal:** JMIR Medical Informatics  
**DOI:** 10.2196/36388  
**Abstract:** The researchers conducted a systematic scoping review examining how racial bias in clinical machine learning models is assessed and mitigated. Searching three databases, they identified 12 relevant studies involving ML applications including diagnosis, outcome prediction, and clinical score prediction. Key findings indicate that 67% of studies identified racial bias present, while mitigation strategies successfully improved fairness metrics. The most common fairness measures included "equal opportunity difference (5/12, 42%), accuracy (4/12, 25%), and disparate impact (2/12, 17%)." Preprocessing methods were most frequently employed for bias mitigation. The authors conclude that "standardized reporting and data availability in medical ML studies" would enhance transparency and facilitate bias evaluation.

---

## Paper 8

**PMID:** 39055787  
**Title:** Stakeholder perspectives on ethical and trustworthy voice AI in health care  
**Authors:** Jean-Christophe Bélisle-Pipon, Maria Powell, Renee English, Marie-Françoise Malo, Vardit Ravitsky, and the Bridge2AI–Voice Consortium (with Yael Bensoussan)  
**Year:** 2024  
**Journal:** Digital Health  
**DOI:** 10.1177/20552076241260407  
**Abstract:** This study surveyed 27 stakeholders — including voice AI experts, clinicians, scholars, patients, trainees, and policy-makers — from the 2023 Voice AI Symposium. Researchers sought to understand perspectives on developing ethical and trustworthy voice AI systems for healthcare applications. Key findings identified priorities regarding ethical concerns, established criteria for "ethically sourced data," explored synthetic voice data applications, and proposed frameworks for ensuring trustworthiness. The study represents "the first stakeholder survey related to voice as a biomarker of health published to date," highlighting how voice technology through smartphones and telehealth could address health disparities while emphasizing the critical need for bias-free datasets and ethical innovation in voice AI development.

---

## Paper 9

**PMID:** 39173183  
**Title:** Health Equity and Ethical Considerations in Using Artificial Intelligence in Public Health and Medicine  
**Authors:** Irene Dankwa-Mullan  
**Year:** 2024  
**Journal:** Preventing Chronic Disease  
**DOI:** 10.5888/pcd21.240245  
**Abstract:** This commentary examines how health equity and ethics shape AI implementation in public health and medicine. The piece addresses the tension between AI's potential benefits and risks, noting that deployment could amplify existing health disparities. The author emphasizes "ethical social responsibility" and explores implications for practice and policy, ultimately offering guidance on leveraging AI advancements responsibly while maintaining equitable outcomes across all populations. Concerns are identified that AI may worsen disparities, especially for racial and ethnic minorities, individuals with disabilities, and low-income populations, who often face a higher disease burden.

---

## Paper 10

**PMID:** 34811466  
**Title:** Advancing health equity with artificial intelligence  
**Authors:** Nicole M Thomasian, Carsten Eickhoff, Eli Y Adashi  
**Year:** 2021  
**Journal:** Journal of Public Health Policy  
**DOI:** 10.1057/s41271-021-00319-5  
**Abstract:** The authors examine how AI technologies can both advance and undermine health equity. They argue that "if the benefits of emerging AI technologies are to be realized, consensus around the regulation of algorithmic bias at the policy level is needed." The paper uses historical and hypothetical examples to demonstrate how biases embedded in AI systems can perpetuate healthcare disparities. The authors propose three regulatory oversight principles addressing different phases of the algorithm lifecycle to mitigate bias and ensure ethical integration of AI into healthcare systems.

---

## Paper 11

**PMID:** 38955956  
**Title:** Bridging Health Disparities in the Data-Driven World of Artificial Intelligence: A Narrative Review  
**Authors:** Anastasia Murphy, Kuan Bowen, Isaam M El Naqa, Balaurunathan Yoga, B Lee Green  
**Year:** 2024  
**Journal:** Journal of Racial and Ethnic Health Disparities  
**DOI:** 10.1007/s40615-024-02057-2  
**Abstract:** This narrative review examines whether AI can reduce health disparities or may worsen them. The researchers searched MEDLINE for publications discussing AI's impact on racial/ethnic health disparities in the U.S. Among 65 articles reviewed, they identified six key limitations: "biases in AI can perpetuate and exacerbate racial and ethnic inequities"; the need for algorithmic equity; insufficient diversity in AI fields; requirements for regulation and testing; necessity of ethical standards; and importance of transparency and accountability. The authors conclude that while AI offers promise, "it must be approached with an equity lens during all phases of development."

---

## Paper 12

**PMID:** 41163810  
**Title:** Gender and racial bias unveiled: clinical artificial intelligence (AI) and machine learning (ML) algorithms are fanning the flames of inequity  
**Authors:** Ahmed Umar Otokiti, Huan-Ju Shih, Karmen S Williams  
**Year:** 2025  
**Journal:** Oxford Open Digital Health  
**DOI:** 10.1093/oodh/oqaf027  
**Abstract:** This systematic review examined whether published clinical AI/ML studies report demographic data on their training datasets. Researchers searched six databases for studies with direct patient care implications, ultimately analyzing 390 publications. Key findings revealed significant transparency gaps: "84% of global models did not report the racial composition of their training data, while 31% lacked gender data." US-based models performed somewhat better, with 56% reporting race and 77% reporting gender. Additionally, "only 16% of all models utilized publicly available, non-proprietary datasets." The authors conclude that "standardized reporting of gender and racial composition in training data is urgently needed" to ensure ethical deployment of these technologies and address health equity concerns.

---

## Paper 13

**PMID:** 37661144  
**Title:** Guess What We Can Hear — Novel Voice Biomarkers for the Remote Detection of Disease  
**Authors:** Jaskanwal Deep Singh Sara, Diana Orbelo, Elad Maor, Lilach O Lerman, Amir Lerman  
**Year:** 2023  
**Journal:** Mayo Clinic Proceedings  
**DOI:** 10.1016/j.mayocp.2023.03.007  
**Abstract:** This review examines how voice analysis combined with artificial intelligence can support telemedicine. The authors describe "voice biomarkers, obtained from the extraction of characteristic acoustic and linguistic features, are associated with a variety of diseases" including COVID-19. The work presents a classification framework for voice biomarkers, discusses potential biological mechanisms linking vocal characteristics to disease states, and reviews evidence connecting voice features to cardiovascular, neurological, psychiatric, and infectious conditions. The authors outline the development process from recording samples through machine learning algorithm training and emphasize the importance of clinical trials, data security, and privacy protections in this emerging field.

---

## Paper 14

**PMID:** 37203624  
**Title:** Classification of Parkinson's Disease from Voice — Analysis of Data Selection Bias  
**Authors:** Alexander Brenner, Catharina Marie Van Alen, Lucas Plagwitz, Julian Varghese  
**Year:** 2023  
**Journal:** Studies in Health Technology and Informatics  
**DOI:** 10.3233/SHTI230079  
**Abstract:** The research examines how machine learning models can identify Parkinson's disease through voice analysis using the mPower study database. The authors note that "the dataset has unbalanced class, gender and age distribution," which necessitates careful sampling strategies. Their work focuses on detecting biases such as identity confounding and inadvertent learning of characteristics unrelated to disease (non-disease-specific characteristics), proposing sampling techniques to mitigate these methodological challenges. This paper directly addresses how demographic imbalances in voice datasets introduce bias in disease classification models.

---

## Paper 15

**PMID:** 38929263  
**Title:** Voice as a Biomarker of Pediatric Health: A Scoping Review  
**Authors:** Hannah Paige Rogers, Anne Hseu, Jung Kim, Elizabeth Silberholz, Stacy Jo, Anna Dorste, Kathy Jenkins; Bridge2AI-Voice Consortium  
**Year:** 2024  
**Journal:** Children (Basel)  
**DOI:** 10.3390/children11060684  
**Abstract:** This scoping review examines how artificial intelligence can analyze children's voices (ages 0–17) as health biomarkers. The analysis synthesized 62 studies using feature extraction and AI models to detect pathological indicators. The research identified autism spectrum disorder, intellectual disabilities, asphyxia, and asthma as the most frequently studied conditions. Mel-Frequency Cepstral Coefficients and Support Vector Machines emerged as predominant methodologies. The findings suggest that "voice analysis using AI demonstrates promise as a non-invasive, cost-effective biomarker" across pediatric conditions, though standardization of methods is needed for clinical implementation.

---

## Paper 16

**PMID:** 38386315  
**Title:** Voice as an AI Biomarker of Health — Introducing Audiomics  
**Authors:** Yaël Bensoussan, Olivier Elemento, Anaïs Rameau  
**Year:** 2024  
**Journal:** JAMA Otolaryngology – Head & Neck Surgery  
**DOI:** 10.1001/jamaoto.2023.4807  
**Abstract:** This Viewpoint discusses the need to create standards for audiomics — the systematic identification of unique audio biomarkers of health and disease — now possible because of more efficient voice data analysis available through the use of artificial intelligence (AI). The authors argue that standardization of audiomics methodology is essential for improving patient care and enabling reproducible, equitable clinical applications of voice AI. The piece calls for community-wide consensus on data collection, feature extraction, and reporting to ensure voice biomarkers can be validated across diverse populations.

---

## Paper 17

**PMID:** 32457147  
**Title:** Gender imbalance in medical imaging datasets produces biased classifiers for computer-aided diagnosis  
**Authors:** Agostina J Larrazabal, Nicolás Nieto, Victoria Peterson, Diego H Milone, Enzo Ferrante  
**Year:** 2020  
**Journal:** Proceedings of the National Academy of Sciences USA  
**DOI:** 10.1073/pnas.1919012117  
**Abstract:** The research demonstrates that imbalanced gender representation in medical imaging datasets negatively impacts AI diagnostic performance. The team trained three deep neural network architectures on two public X-ray datasets to diagnose thoracic diseases under varying gender imbalance conditions. Their findings revealed "a consistent decrease in performance for underrepresented genders when a minimum balance is not fulfilled." The authors conclude that regulatory agencies should establish explicit gender balance recommendations for computer-assisted diagnosis systems, and the medical imaging community must develop algorithms robust against such imbalances. The findings generalize directly to voice/audio dataset design for health AI.

---

## Paper 18

**PMID:** 32529043  
**Title:** Sex and gender differences and biases in artificial intelligence for biomedicine and healthcare  
**Authors:** Davide Cirillo, Silvina Catuara-Solarz, Czuee Morey, Emre Guney, Laia Subirats, Simona Mellino, Annalisa Gigante, Alfonso Valencia, María José Rementeria, Antonella Santuccione Chadha, Nikolaos Mavridis  
**Year:** 2020  
**Journal:** NPJ Digital Medicine  
**DOI:** 10.1038/s41746-020-0288-5  
**Abstract:** The authors examine how artificial intelligence technologies in precision medicine often fail to account for sex and gender dimensions in health and disease. They note that "most of the currently used biomedical AI technologies do not account for bias detection" and that algorithmic design typically overlooks these critical variables. The review identifies current gaps in biomedical AI applications and offers recommendations to improve outcomes, reduce health inequalities, and optimize AI utilization across diverse populations while accounting for individual differences rooted in genetic and environmental factors. The majority of reviewed algorithms ignore the sex and gender dimension and its contribution to health and disease differences among individuals.

---

## Paper 19

**PMID:** 32205437  
**Title:** Racial disparities in automated speech recognition  
**Authors:** Allison Koenecke, Andrew Nam, Emily Lake, Joe Nudell, Minnie Quartey, Zion Mengesha, Connor Toups, John R Rickford, Dan Jurafsky, Sharad Goel  
**Year:** 2020  
**Journal:** Proceedings of the National Academy of Sciences USA  
**DOI:** 10.1073/pnas.1915768117  
**Abstract:** The research examines five major speech recognition systems from Amazon, Apple, Google, IBM, and Microsoft. Using audio from 42 white and 73 Black speakers across five U.S. cities, the study found significant performance gaps. The authors report that "all five ASR systems exhibited substantial racial disparities, with an average word error rate (WER) of 0.35 for black speakers compared with 0.19 for white speakers." They traced these disparities to acoustic models and propose addressing them through more diverse training datasets, including African American Vernacular English representations. This landmark study is directly relevant to equity in any speech-based health AI system.

---

## Paper 20

**PMID:** 34383925  
**Title:** Bias and fairness assessment of a natural language processing opioid misuse classifier: detection and mitigation of electronic health record data disadvantages across racial subgroups  
**Authors:** Hale M Thompson, Brihat Sharma, Sameer Bhalla, Randy Boley, Connor McCluskey, Dmitriy Dligach, Matthew M Churpek, Niranjan S Karnik, Majid Afshar  
**Year:** 2021  
**Journal:** Journal of the American Medical Informatics Association (JAMIA)  
**DOI:** 10.1093/jamia/ocab148  
**Abstract:** This research evaluated fairness and bias in a machine learning classifier for detecting opioid misuse using electronic health records. The study analyzed two datasets (original: n=1,000; external validation: n=53,974) from two health systems across racial/ethnic groups (Black, Hispanic/Latinx, White, Other). Key findings included disparities in false negative rates between groups: "the Black subgroup compared to the FNR (0.17) of the White subgroup" showed significantly higher misses. Similar predictive features like "heroin" and "substance abuse" appeared across groups, yet inequities persisted. The researchers successfully implemented post-hoc recalibration techniques that "eliminated bias in FNR with minimal changes in other subgroup error metrics." The study concluded that "standardized, transparent bias assessments are needed to improve trustworthiness in clinical machine learning models."

---

## Summary Table

| # | PMID | Title (short) | Year | Theme |
|---|------|---------------|------|-------|
| 1 | 37380750 | Algorithmic fairness in AI for medicine | 2023 | Fairness/bias in health AI |
| 2 | 36084616 | Algorithmic fairness in computational medicine | 2022 | Fairness measurement & mitigation |
| 3 | 37615031 | Fairness + bias mitigation via deep RL | 2023 | Bias mitigation methods |
| 4 | 36060496 | Enabling fairness in healthcare through ML | 2022 | Affirmative algorithms / equity |
| 5 | 33981989 | Fairness, bias, AI/ML in global health | 2021 | Global health / LMIC equity |
| 6 | 38677633 | Survey: AI fairness & bias methods in biomedicine | 2024 | Comprehensive debiasing survey |
| 7 | 35639450 | Racial bias in clinical ML models: scoping review | 2022 | Racial bias mitigation |
| 8 | 39055787 | Stakeholder perspectives on ethical voice AI | 2024 | Voice AI ethics / bias-free data |
| 9 | 39173183 | Health equity & ethics in AI / public health | 2024 | Health equity policy |
| 10 | 34811466 | Advancing health equity with AI | 2021 | Regulatory frameworks |
| 11 | 38955956 | Bridging health disparities in AI (narrative review) | 2024 | Racial/ethnic disparities |
| 12 | 41163810 | Gender & racial bias in clinical AI/ML | 2025 | Training data transparency |
| 13 | 37661144 | Novel voice biomarkers for remote disease detection | 2023 | Voice biomarkers / telemedicine |
| 14 | 37203624 | Parkinson's from voice — data selection bias | 2023 | Voice ML bias / dataset balance |
| 15 | 38929263 | Voice biomarker of pediatric health: scoping review | 2024 | Pediatric voice AI |
| 16 | 38386315 | Voice as AI biomarker — introducing audiomics | 2024 | Audiomics standards |
| 17 | 32457147 | Gender imbalance in medical imaging → biased classifiers | 2020 | Dataset diversity / sex bias |
| 18 | 32529043 | Sex/gender differences & biases in biomedical AI | 2020 | Sex/gender bias in AI |
| 19 | 32205437 | Racial disparities in automated speech recognition | 2020 | ASR racial bias — core relevance |
| 20 | 34383925 | Bias in NLP opioid misuse classifier (EHR/racial) | 2021 | NLP fairness / clinical ML |

---

*Sources retrieved from pubmed.ncbi.nlm.nih.gov. All PMIDs verified.*
