## Automatic Diagnosis

This use case demonstrates how to automatically add a diagnosis to a PathoML representation based on a diagnostic criterion.

#### Important Files

PathoML.owl: Ontology specification of PathoML.

Tumor Pathology Ontology Simplified.owl: The logic of the diagnostic criterion is represented in Tumor Pathology Ontology .

Diagnosing.java: Source code of automatic diagnosis.

Automatic Subtyping.xml: The input PathoML representation of a slide with several histopathological phenotypes detected while lacking a subtyping diagnosis.

Automatic Subtyping Diagnosis.xml: The output PathoML representation of the slide which involves the generated diagnostic result.

#### How to Use

Run the use case:

```shell
java -jar Diagnosing.jar "Automatic Subtyping.xml"
```

Java 11 runtime is required.

#### Example Output

Below is an example output of running the above code:

> Diagnosis is Human Papillomavirus-Related Endocervical Adenocarcinoma
>
> * identifier: 9283245040, title: Female Genital Tumours: WHO Classification of Tumours (Medicine) 5th Edition
> * quotation: Hallmarks of HPV-associated endocervical adenocarcinoma architecture include apical mitoses and karyorrhexis, conspicuous and identifiable at low-power magnification.
> * evidences: Glandular_Pattern5 Glandular_Pattern4
>
> Diagnosis is Adenocarcinoma, HPV-associated, usual type
>
> * identifier: 9283245040, title: Female Genital Tumours: WHO Classification of Tumours (Medicine) 5th Edition
> * quotation: Cells with mucinous cytoplasm constitute only 0-50% of the tumour.
> * evidences: Quantitative_Metric1

The output is an illustration of a two-step reasoning process of ontology reasoner. 
  1. Firstly, the reasoner subtypes the slide as **Human Papillomavirus-Related Endocervical Adenocarcinoma** whose evidences are the two histopathological phenotypes of the slide named "**Glandular_Pattern4**" and "**Glandular_Pattern5**" in Automatic Subtyping.xml; the official guideline on which the first step of reasoning is based is **Female Genital Tumours: WHO Classification of Tumours (Medicine) 5th Edition**, its isbn number is **9283245040** and the quotation is "Hallmarks of HPV-associated endocervical adenocarcinoma architecture include apical mitoses and karyorrhexis, conspicuous and identifiable at low-power magnification". 
  2. Based on the diagnostic result of the first step as well as **Quantitative_Metric1**, the reasoner further subtypes the slide as **HPV-associated, usual type** with the evidences and citation of the guideline provided as well.

#### Why using a simplified Tumor Pathology Ontology instead of a complete one?
Reasoning over a simplified Tumor Pathology Ontology which contains only the necessary classes and axioms for automatic diagnosis is more time-efficient than a complete Tumor Pathology Ontology through ontology reasoner.

