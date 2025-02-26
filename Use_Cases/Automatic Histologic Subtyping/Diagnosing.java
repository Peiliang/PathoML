import org.apache.commons.io.FileUtils;
import org.semanticweb.HermiT.ReasonerFactory;
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.formats.RDFXMLDocumentFormat;
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.reasoner.NodeSet;
import org.semanticweb.owlapi.reasoner.OWLReasoner;
import org.semanticweb.owlapi.reasoner.OWLReasonerFactory;
import org.semanticweb.owlapi.util.OWLOntologyMerger;
import org.semanticweb.owlapi.vocab.OWL2Datatype;
import org.w3c.dom.Document;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Deduction {
    public static OWLOntologyManager manager, histoMLManager;
    public static OWLOntology histoML, histo, diagnosis, merged;
    public static OWLDataFactory dataFactory;
    public static IRI diagnosisIRI = IRI.create("http://www.aipath.com/histo/diagnosis.owl");
    public static IRI histoMLIRI = IRI.create("https://pathoml-1308125782.cos.ap-chengdu.myqcloud.com/PathoML.owl");
    public static IRI mergedIRI = IRI.create("http://www.aipath.com/histo/merged.owl");
    public static OWLReasonerFactory reasonerFactory;
    public static OWLReasoner histoReasoner, histoMLReasoner, mergedReasoner;
    public static OWLClass diagnosisClazz, diseaseClazz;
    public static Map<OWLClass, OWLClass> diagnosisResults, diseaseMapping;
    public static RDFXMLDocumentFormat format = new RDFXMLDocumentFormat();
    public static DocumentBuilderFactory builderFactory;
    public static DocumentBuilder builder;
    public static OWLAnnotationProperty identifierProp, definitionProp, titleProp;
    public static DiseaseNode diseaseRoot;
    public static Map<OWLClass, OWLClass> evidenceClasses = new HashMap<>();
    public static Map<OWLClass, OWLObjectProperty> evidenceProperties = new HashMap<>();
    public static Map<OWLClass, OWLClass> evidences;
    public static List<String> diagnosisString = new ArrayList<>();

    public static class DiseaseNode {
        public long identifier;
        public String definition = "", title = "";
        public OWLClass diseaseClazz;
        public List<OWLClassExpression> equivalentList = new ArrayList<>();
        public List<DiseaseNode> children = new ArrayList<>();

        public DiseaseNode(OWLClass clazz, OWLReasoner reasoner) {
            diseaseClazz = clazz;
            histo.getEquivalentClassesAxioms(clazz).stream().findFirst().ifPresent(axiom -> {
                axiom.annotations(identifierProp).findFirst().flatMap(annotation
                        -> annotation.getValue().asLiteral()).ifPresent(literal -> identifier = Long.parseLong(literal.getLiteral()));
                axiom.annotations(definitionProp).findFirst().flatMap(annotation
                        -> annotation.getValue().asLiteral()).ifPresent(literal -> definition = literal.getLiteral());
                axiom.annotations(titleProp).findFirst().flatMap(annotation
                        -> annotation.getValue().asLiteral()).ifPresent(literal -> title = literal.getLiteral());

                axiom.classExpressions().forEach(intersectionClazz -> intersectionClazz.conjunctSet().forEach(equivalentClazz -> {
                    if (equivalentClazz.getClassExpressionType() == ClassExpressionType.OBJECT_SOME_VALUES_FROM) {
                        equivalentList.add(((OWLObjectSomeValuesFrom) equivalentClazz).getFiller());
                    }
                }));
            });

            for (OWLClass child : reasoner.getSubClasses(clazz, true).getFlattened())
                if (!child.isOWLNothing())
                    children.add(new DiseaseNode(child, reasoner));
        }
    }

    public static void main(String[] args) throws Exception {
        histoMLManager = OWLManager.createOWLOntologyManager();
        histoML = histoMLManager.loadOntologyFromOntologyDocument(new File("PathoML.owl"));
        manager = OWLManager.createOWLOntologyManager();
        histo = manager.loadOntologyFromOntologyDocument(new File("Tumor Pathology Ontology Simplified.owl"));
        dataFactory = manager.getOWLDataFactory();
        diagnosisClazz = dataFactory.getOWLClass("https://pathoml-1308125782.cos.ap-chengdu.myqcloud.com/PathoML.owl#Diagnosis");
        diseaseClazz = dataFactory.getOWLClass("http://www.aipath.com/histo/TPO_0000003");
        identifierProp = dataFactory.getOWLAnnotationProperty("http://purl.org/dc/terms/identifier");
        definitionProp = dataFactory.getOWLAnnotationProperty("http://purl.obolibrary.org/obo/IAO_0000115");
        titleProp = dataFactory.getOWLAnnotationProperty("http://purl.org/dc/terms/title");

        builderFactory = DocumentBuilderFactory.newInstance();
        builder = builderFactory.newDocumentBuilder();

        format.setPrefix("histo", "http://www.aipath.com/histo/histo.owl#");

        reasonerFactory = new ReasonerFactory();
        histoReasoner = reasonerFactory.createReasoner(histo);
        histoMLReasoner = reasonerFactory.createReasoner(histoML);
        diseaseRoot = new DiseaseNode(diseaseClazz, histoReasoner);

        OWLClass phenotype = dataFactory.getOWLClass("https://pathoml-1308125782.cos.ap-chengdu.myqcloud.com/PathoML.owl#Phenotype");
        histoMLReasoner.getSubClasses(phenotype).entities().forEach(evidenceClazz -> {
            evidenceProperties.put(evidenceClazz, dataFactory.getOWLObjectProperty("http://www.aipath.com/histo/HISTO_0000045"));
            evidenceClasses.put(evidenceClazz, dataFactory.getOWLClass("http://purl.obolibrary.org/obo/NCIT_C83490"));
        });
        OWLClass quantitative = dataFactory.getOWLClass("https://pathoml-1308125782.cos.ap-chengdu.myqcloud.com/PathoML.owl#Quantitative_Metric");
        evidenceProperties.put(quantitative, dataFactory.getOWLObjectProperty("http://www.aipath.com/histo/HISTO_0000048"));
        evidenceClasses.put(quantitative, dataFactory.getOWLClass("http://www.aipath.com/histo/HISTO_0000057"));

        processDiagnosis(args[0]);
    }

    public static void processDiagnosis(String filename) throws Exception {
        diagnosis = manager.createOntology(diagnosisIRI);
        diagnosisResults = new HashMap<>();
        diseaseMapping = new HashMap<>();
        evidences = new HashMap<>();

        try (InputStream is = new FileInputStream(filename)) {
            Document doc = builder.parse(is);

            Node root = doc.getChildNodes().item(0);
            if (root.hasChildNodes()) {
                NodeList nodeList = root.getChildNodes();
                for (int count = 0; count < nodeList.getLength(); count++) {
                    Node node = nodeList.item(count);

                    if (node.getNodeType() == Node.ELEMENT_NODE) {
                        processNode(node);
                    }
                }
            }
        }

        OWLClass diagnosisClazz = null;
        if (diagnosisResults.isEmpty()) {
            diagnosisClazz = dataFactory.getOWLClass(diagnosisIRI + "#Diagnosis");
            List<OWLClassExpression> equivalentList = new ArrayList<>();
            equivalentList.add(dataFactory.getOWLClass("http://www.aipath.com/histo/histo.owl#Diagnosis"));

            for (Map.Entry<OWLClass, OWLClass> pair : evidences.entrySet()) {
                OWLObjectProperty property = evidenceProperties.get(pair.getValue());
                equivalentList.add(dataFactory.getOWLObjectSomeValuesFrom(property, pair.getKey()));
            }

            diagnosis.add(dataFactory.getOWLEquivalentClassesAxiom(diagnosisClazz, dataFactory.getOWLObjectIntersectionOf(equivalentList)));
        }

        OWLOntologyMerger merger = new OWLOntologyMerger(manager);
        merged = merger.createMergedOntology(manager, mergedIRI);
//        manager.saveOntology(diagnosis, format, new FileOutputStream("diagnosis.owl"));
//        manager.saveOntology(merged, format, new FileOutputStream("test.owl"));

        mergedReasoner = reasonerFactory.createReasoner(merged);
        if (diagnosisResults.isEmpty()) {
            diagnosisString.clear();
            OWLClass diagnosis = dataFactory.getOWLClass("http://www.aipath.com/histo/histo.owl#Diagnosis");
            OWLClass uniXref = dataFactory.getOWLClass("http://www.aipath.com/histo/histo.owl#UnificationXref");

            System.out.println("The diagnosis process of ontology reasoner:");
            List<DiseaseNode> diseases = searchDisease(diagnosisClazz);
            OWLClass diseaseClass = diseases.get(diseases.size() - 1).diseaseClazz;
            OWLClass XrefClass = dataFactory.getOWLClass(diagnosisIRI + "#UnificationXref_Diagnosis");

            writeXMLNode(diagnosis, diagnosisClazz, true);
            writeXMLNode(dataFactory.getOWLObjectProperty("http://www.aipath.com/histo/histo.owl#hasXref"), XrefClass, false);
            for (Map.Entry<OWLClass, OWLClass> pair : evidences.entrySet())
                writeXMLNode(evidenceProperties.get(pair.getValue()), pair.getKey(), false);
            writeXMLEnd(diagnosis);

            writeXMLNode(uniXref, XrefClass, true);
            writeXMLDataNode(dataFactory.getOWLObjectProperty("http://www.aipath.com/histo/histo.owl#uri"), diseaseClass.toStringID());
            writeXMLEnd(uniXref);

            String content = FileUtils.readFileToString(new File(filename), "utf-8");
            int index = content.indexOf("</rdf:RDF>");
            StringBuilder builder = new StringBuilder();
            builder.append(content, 0, index);
            for (String str : diagnosisString) {
                builder.append(str);
                builder.append("\n");
            }
            builder.append(content.substring(index));
            FileUtils.writeStringToFile(new File(filename.replace(".xml", " Diagnosis.xml")), builder.toString(), "utf-8");
        } else {
            for (OWLClass owlClazz : diagnosisResults.keySet()) {
                OWLClass typeClazz = diagnosisResults.get(owlClazz);
                OWLClass diseaseClass = diseaseMapping.get(typeClazz);
                diagnosisResults.put(owlClazz, diseaseClass);
                boolean right = mergedReasoner.getSuperClasses(owlClazz).containsEntity(diseaseClass);
                System.out.printf("%s is %s : %s\n", getLabel(owlClazz), getLabel(diseaseClass), right);
                System.out.println("-----------------------------------------------------------------");
                System.out.println("The diagnosis process of ontology reasoner:");
                searchDisease(owlClazz);
                System.out.println();
            }
        }

        manager.removeOntology(diagnosis);
        manager.removeOntology(merged);
        System.out.println();
    }

    public static void processNode(Node node) {
        String name = node.getNodeName();
        if (name.startsWith("histo:")) {
            OWLClass baseClazz = dataFactory.getOWLClass(name, format);

            IRI nodeIRI = processAttributes(node.getAttributes());
            OWLClass owlClazz = dataFactory.getOWLClass(nodeIRI);

            List<OWLClassExpression> equivalentList = new ArrayList<>();
            equivalentList.add(baseClazz);

            OWLClass histoMLClazz = dataFactory.getOWLClass(name.replace("histo:", histoMLIRI + "#"));
            if (histoMLReasoner.getSuperClasses(histoMLClazz).containsEntity(diagnosisClazz))
                diagnosisResults.put(owlClazz, null);

            if (evidenceClasses.containsKey(histoMLClazz))
                evidences.put(owlClazz, histoMLClazz);

            if (node.hasChildNodes()) {
                NodeList nodeList = node.getChildNodes();

                for (int i = 0; i < nodeList.getLength(); i++) {
                    Node childNode = nodeList.item(i);

                    if (childNode.getNodeType() == Node.ELEMENT_NODE) {
                        if (childNode.getNodeName().equals("histo:has_Diagnosis_Evidence")) {
                            IRI parentNodeIRI = processAttributes(childNode.getAttributes());
                            OWLClass parentClazz = dataFactory.getOWLClass(parentNodeIRI);

                            equivalentList.add(parentClazz);
                        }

                        if (childNode.hasAttributes()) {
                            IRI childNodeIRI = processAttributes(childNode.getAttributes());
                            if (childNodeIRI != null) {
                                OWLClass childClazz = dataFactory.getOWLClass(childNodeIRI);

                                if (childNode.getNodeName().equals("histo:hasXref")) {
                                    if (diagnosisResults.containsKey(owlClazz))
                                        diagnosisResults.put(owlClazz, childClazz);
                                    else equivalentList.add(childClazz);
                                } else if (childNode.getNodeName().equals("histo:has_Morphologic_Evidence")) {
                                    OWLObjectProperty property = dataFactory.getOWLObjectProperty("http://www.aipath.com/histo/HISTO_0000045");
                                    equivalentList.add(dataFactory.getOWLObjectSomeValuesFrom(property, childClazz));
                                } else if (childNode.getNodeName().equals("histo:has_Quantitative_Metric_Evidence")) {
                                    OWLObjectProperty property = dataFactory.getOWLObjectProperty("http://www.aipath.com/histo/HISTO_0000048");
                                    equivalentList.add(dataFactory.getOWLObjectSomeValuesFrom(property, childClazz));
                                }
                            } else {
                                String value = childNode.getTextContent();
                                OWLDataProperty property = dataFactory.getOWLDataProperty(childNode.getNodeName(), format);

                                if (childNode.getNodeName().equals("histo:hasValue")) {
                                    Double.parseDouble(value);
                                    OWLDataProperty dataProperty = dataFactory.getOWLDataProperty("http://www.aipath.com/histo/HISTO_0000058");
                                    OWLDatatype decimal = OWL2Datatype.XSD_FLOAT.getDatatype(dataFactory);
                                    equivalentList.add(dataFactory.getOWLDataHasValue(dataProperty, dataFactory.getOWLLiteral(value, decimal)));
                                } else if (childNode.getNodeName().equals("histo:uri")) {
                                    OWLClass parentClazz = dataFactory.getOWLClass(value);
                                    equivalentList.add(parentClazz);
                                    diseaseMapping.put(owlClazz, parentClazz);
                                } else {
                                    equivalentList.add(dataFactory.getOWLDataHasValue(property, dataFactory.getOWLLiteral(value)));
                                }
                            }
                        }
                    }
                }
            }

            diagnosis.add(dataFactory.getOWLEquivalentClassesAxiom(owlClazz, dataFactory.getOWLObjectIntersectionOf(equivalentList)));
        }
    }

    public static IRI processAttributes(NamedNodeMap attributes) {
        for (int i = 0; i < attributes.getLength(); i++) {
            Node attribute = attributes.item(i);

            String name = attribute.getNodeName();
            String value = attribute.getNodeValue();

            if (name.equals("rdf:ID")) return IRI.create(diagnosisIRI + "#" + value);
            if (name.equals("rdf:resource")) return IRI.create(diagnosisIRI + value);
            if (name.equals("rdf:datatype")) return null;
        }

        throw new RuntimeException("Unable to parse attributes");
    }

    public static List<DiseaseNode> searchDisease(OWLClass owlClass) {
        NodeSet<OWLClass> superClasses = mergedReasoner.getSuperClasses(owlClass);
        List<DiseaseNode> diseases = new ArrayList<>();
        searchDiseaseRecursive(diseaseRoot, owlClass, superClasses, diseases);
        return diseases;
    }

    public static void searchDiseaseRecursive(DiseaseNode disease, OWLClass owlClass, NodeSet<OWLClass> superClasses, List<DiseaseNode> diseases) {
        if (disease.identifier != 0 && superClasses.containsEntity(disease.diseaseClazz)) {
            System.out.printf("%s is %s\n", getLabel(owlClass), getLabel(disease.diseaseClazz));
            System.out.printf("\tidentifier: %d, title: %s\n", disease.identifier, disease.title);
            System.out.printf("\tquotation: %s\n", disease.definition);
            diseases.add(disease);
            if (!evidences.isEmpty()) {
                System.out.print("\tevidences: ");
                for (OWLClassExpression equivalentClazz : disease.equivalentList) {
                    NodeSet<OWLClass> subClasses = mergedReasoner.getSubClasses(equivalentClazz);
                    for (OWLClass evidenceClazz : evidences.keySet()) {
                        if (subClasses.containsEntity(evidenceClazz)) {
                            System.out.printf("%s ", getLabel(evidenceClazz));
                            break;
                        }
                    }
                }
                System.out.println();
            }
        }

        for (DiseaseNode child : disease.children)
            searchDiseaseRecursive(child, owlClass, superClasses, diseases);
    }

    public static String getClassName(OWLEntity type) {
        String name = type.toStringID();
        int index = name.indexOf('#');
        return index >= 0 ? name.substring(index + 1) : name;
    }

    public static void writeXMLNode(OWLEntity type, OWLClass clazz, boolean main) {
        diagnosisString.add(String.format("%s<histo:%s rdf:%s=\"%s%s\"%s>", main ? "\t" : "\t\t", getLabel(type),
                main ? "ID" : "resource", main ? "" : "#", getClassName(clazz), main ? "" : "/"));
    }

    public static void writeXMLDataNode(OWLEntity type, String data) {
        diagnosisString.add(String.format("\t\t<histo:%s>%s</histo:%s>", getClassName(type), data, getClassName(type)));
    }

    public static void writeXMLEnd(OWLEntity type) {
        diagnosisString.add(String.format("\t</histo:%s>", getClassName(type)));
    }

    public static String getLabel(OWLEntity type) {
        for (OWLAnnotationAssertionAxiom axiom : histo.getAnnotationAssertionAxioms(type.getIRI())) {
            if (axiom.getProperty().isLabel()) {
                if (axiom.getValue() instanceof OWLLiteral) {
                    OWLLiteral val = (OWLLiteral) axiom.getValue();
                    return val.getLiteral();
                }
            }
        }
        return getClassName(type);
    }
}
