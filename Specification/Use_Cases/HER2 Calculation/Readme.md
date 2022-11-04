## HER2 Phenotype Characterization

This use case demonstrate how to calculate HER2 using HistoML.

#### Important Files

* HER2_Calculation.xml: HistoML representation file of a HER2 phenotype.
* Stroma.rq and Lymphocytes.rq: Sparql queries for stroma components and lymphocytes.
* Annotation.db: Area annotation of the cell membranes and nucleolus.
* CalculateHER2.java: Java code to execute Sparql queries and calculate HER2.
* Execute.py: Execute the use case.

#### How to Use

Install all requirements for python:

```shell
pip3 install -r requirements.txt
```

Run the use case:

```shell
python3 Execute.py
```

Java 11 runtime is required to run the queries.
