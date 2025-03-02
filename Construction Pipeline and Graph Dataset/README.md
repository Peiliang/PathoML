# PathoML OWL File Structure

## 1. OWL Ontology Declaration
- **Ontology Node**
  - Declared as `owl:Ontology`
  - Contains an `owl:imports` statement pointing to the remote PathoML ontology (`https://pathoml-1308125782.cos.ap-chengdu.myqcloud.com/PathoML.owl`)

## 2. Provenance (Data Source)
- **Instance:** `Provenance`
  - `rdf:type` → `patho:Provenance`
  - `patho:comment` → Contains `website` and `dataset_description` as data source information

## 3. Patch (HE Image Patch)
- **Instance:** `patch`
  - `rdf:type` → `patho:HE_Patch`
  - `patho:displayName` → Patch name
  - `patho:availability` → Patch file location (file path)

## 4. UnificationXref (Standard References)
- **Instances:** Five key references related to pathological features:
  - `UnificationXref_Tumor` → `NCIT_C18009` (Tumor)
  - `UnificationXref_Nucleus` → `GO_0005634` (Nucleus)
  - `UnificationXref_Size` → `PATO_0002057` (Size)
  - `UnificationXref_Area` → `PATO_0001323` (Area)
  - `UnificationXref_Shape` → `PATO_0005020` (Shape)

- **Properties for each Xref**
  - `rdf:type` → `patho:UnificationXref`
  - `patho:uri` → Corresponding ontology reference (OBO, NCIT, etc.)

## 5. UnitVocabulary (Unit Terms)
- **Instance:** `square_micrometer`
  - `rdf:type` → `patho:UnitVocabulary`
  - `patho:uri` → `UO_0010001` (Square micrometer unit)

## 6. Tissue Reference
- **Instance:** `TissueRef_{tissue_type}`
  - `rdf:type` → `patho:Other_AnatomicalEntityReference`
  - `patho:displayName` → Tissue type (from `updated_tissues`)
  - `patho:hasXref` → `UnificationXref_Tumor` (Associated with tumor reference)

## 7. Cellular Component Reference (Cell Types)
- **Instance:** `CellularComponentRef_{cell_type}`
  - `rdf:type` → `patho:CellularComponentReference`
  - `patho:displayName` → Cell type (from `processed_nuclei`)
  - `patho:hasXref` → `UnificationXref_Nucleus` (Associated with nucleus reference)

## 8. File Structure Summary
<pre>
Ontology
│
├── Provenance
│   ├── patho:comment (website)
│   ├── patho:comment (dataset_description)
│
├── Patch
│   ├── patho:displayName (patch_name)
│   ├── patho:availability (patch_availability)
│
├── UnificationXref
│   ├── Tumor → NCIT_C18009
│   ├── Nucleus → GO_0005634
│   ├── Size → PATO_0002057
│   ├── Area → PATO_0001323
│   ├── Shape → PATO_0005020
│
├── UnitVocabulary
│   ├── SquareMicrometer → UO_0010001
│
├── Tissue References
│   ├── TissueRef_{type} → Other_AnatomicalEntityReference
│   ├── hasXref → Tumor Reference
│
├── Cellular Component References
│   ├── CellularComponentRef_{type} → CellularComponentReference
│   ├── hasXref → Nucleus Reference
</pre>

## Environment Setup
To install dependencies, run:
```bash
pip install -r requirements.txt
```
## How to Run
Execute the following command to run the inference:
```bash
python inference_examples_RGB_optimized.py --if_folder True --image_path /path/to/images --text_prompt "neoplastic cells"
```
`--if_folder`: Set to `True` if processing a folder of images or `False` if processing a single image.

`--image_path`: Path to the image or folder containing images.

`--text_prompt`: Specify the text prompt for analysis.

## Example OWL File Generated for the Bracs Dataset
A sample OWL file generated for the Bracs dataset is available for download:
[Google Drive Link](https://drive.google.com/file/d/1YCmY4jrca1XLYzlV1z137WF3pYfPShQJ/view?usp=drive_link)

## Citation