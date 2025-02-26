## HER2 Phenotype Characterization

This use case demonstrate how to calculate HER2 using PathoML.

#### Important Files

* HER2_Calculation.xml: PathoML representation file of a slide stained using HER2.
* Cell.rq, Membrane.rq, Nucleus.rq: Sparql queries for cells as well as their membranes and nuclei .
* Annotation.db: Area annotation of the cell membranes and nuclei.
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
