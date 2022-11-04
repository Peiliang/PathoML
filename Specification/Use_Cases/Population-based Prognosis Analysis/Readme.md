## Population-based Phenotype Characterization

This use case demonstrates how to analyze histopathological features contained in whole-slide-images of large patient populations using HistoML.

#### Important Files

Grade1~3.rq & Endothelial.rq: Sparql queries for 4 types of cells.

Analysis.java: Source code of generating HistoML representations and run Sparql queries.

Execute.py: Execute the use case.

Representation folder: HistoML representations used in this use case.

#### How to Use

In this use cases, HistoML representations of ccRCC slides are in "Representation" folder. The original CCRCC Nuclei Grading Dataset can be downloaded from [here](https://dataset.chenli.group/home/ccrcc-grading). This dataset contains 1,000 512x512 images for CCRCC Nuclei Grading and Segmentation tasks.

Run the use case:

```shell
java -jar Analysis.jar area diameter circularity entropy
```

Area, diameter, circularity and entropy are calculated for the nucleolus of each ccRCC tumor cell.

Java 11 runtime is required to run the queries.

#### Example Output

> average area:
> grade 1 (45120): 346.531272
> grade 2 (6407): 566.327923
> grade 3 (2780): 809.073741
> endothelial (16691): 293.694566

This means that the average area of all grade 1 nucleoli (45,102 in total) is 346.5 pixels, etc.

The slides were scanned using a KFBIO scanner at 40x of which the resolution is 0.26 µm/pixel at 40x.

