import org.apache.jena.atlas.io.IndentedWriter;
import org.apache.jena.query.*;
import org.apache.jena.rdf.model.*;
import org.apache.jena.riot.RDFDataMgr;
import org.apache.jena.util.FileUtils;

import java.io.IOException;
import java.util.*;

public class Search {
    public static Model model = ModelFactory.createDefaultModel();
    public static String histoPrefix = "https://histoml-1308125782.cos.ap-chengdu.myqcloud.com/HistoML.owl";
    private static final boolean debug = false;

    public static void her2() throws IOException {
        histoPrefix = "https://pathoml-1308125782.cos.ap-chengdu.myqcloud.com/PathoML.owl";
        class CellProp {
            public final int id, mnx, mny, fpx, fpy, lpx, lpy;
            public final float ctx, cty;
            public final boolean cw;
            public CellProp(String[] args) {
                id = Integer.parseInt(args[0]);
                mnx = Integer.parseInt(args[1]);
                mny = Integer.parseInt(args[2]);
                ctx = Float.parseFloat(args[3]);
                cty = Float.parseFloat(args[4]);
                fpx = Integer.parseInt(args[5]);
                fpy = Integer.parseInt(args[6]);
                lpx = Integer.parseInt(args[7]);
                lpy = Integer.parseInt(args[8]);
                cw = Integer.parseInt(args[9]) > 0;
            }
        }

        model.read(RDFDataMgr.open("HER2_Calculation.xml"), null);
        if (debug) print(model);

        Map<Integer, CellProp> cellProps = new HashMap<>();
        for (String line : FileUtils.readWholeFileAsUTF8("Property.csv").split("\n")) {
            String[] line_arg = line.strip().split(",");
            CellProp cellProp = new CellProp(line_arg);
            cellProps.put(cellProp.id, cellProp);
        }

        List<RDFNode> cellList = execSingleQuery("cells", join(
                "SELECT ?cell WHERE {",
                    "?cell rdf:type histo:NeoplasticCell ;",
                        "histo:entityReference ?refCell .",
                    "?refCell rdf:type histo:CellReference ;",
                        "histo:hasXref ?XrefCell .",
                    "?XrefCell rdf:type histo:UnificationXref ;",
                        "histo:uri \"http://purl.obolibrary.org/obo/FMA_68646\" .",
                "}"));

        for (RDFNode cell : cellList) {
            int membraneSegmentation = Integer.parseInt(execSingleQuery("membrane", join(
                    "SELECT ?segmentation WHERE {",
                        "?cell histo:hasCellularComponent ?membrane .",
                        "?membrane rdf:type histo:NeoplasticCell ;",
                            "histo:entityReference ?refMembrane ;",
                            "histo:segmentation ?segmentation .",
                        "?refMembrane rdf:type histo:CellularComponentReference ;",
                            "histo:hasXref ?XrefMembrane .",
                        "?XrefMembrane rdf:type histo:UnificationXref ;",
                            "histo:uri \"http://purl.obolibrary.org/obo/GO_0044298\" .",
                    "}"), "cell", cell).get(0).toString());

            int nucleusSegmentation = Integer.parseInt(execSingleQuery("nucleus", join(
                    "SELECT ?segmentation WHERE {",
                        "?cell histo:hasCellularComponent ?nucleus .",
                        "?nucleus rdf:type histo:NeoplasticCell ;",
                            "histo:entityReference ?refNucleus ;",
                            "histo:segmentation ?segmentation .",
                        "?refNucleus rdf:type histo:CellularComponentReference ;",
                            "histo:hasXref ?XrefNucleus .",
                        "?XrefNucleus rdf:type histo:UnificationXref ;",
                            "histo:uri \"http://purl.obolibrary.org/obo/GO_0005634\" .",
                    "}"), "cell", cell).get(0).toString());

            CellProp membraneProp = cellProps.get(membraneSegmentation);
            CellProp nucleusProp = cellProps.get(nucleusSegmentation);
            float fpx = membraneProp.fpx + (membraneProp.mnx - nucleusProp.mnx) - nucleusProp.ctx;
            float fpy = membraneProp.fpy + (membraneProp.mny - nucleusProp.mny) - nucleusProp.cty;
            float lpx = membraneProp.lpx + (membraneProp.mnx - nucleusProp.mnx) - nucleusProp.ctx;
            float lpy = membraneProp.lpy + (membraneProp.mny - nucleusProp.mny) - nucleusProp.cty;

            double angle = (Math.atan2(fpy, fpx) - Math.atan2(lpy, lpx)) / Math.PI / 2;
            if (membraneProp.cw) angle = -angle;
            if (angle < 0) angle += 1;
            
            print(cell);
            System.out.printf("%d %d %f\n", membraneSegmentation, nucleusSegmentation, angle);
        }
    }

    public static void main(String[] args) throws Exception {
        her2();
    }

    public static List<List<RDFNode>> execQuery(String subject, String queryString, Object ... binding) {
        queryString = "PREFIX histo: <" + histoPrefix + "#>\n" +
                "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n" + queryString;

        if (debug) System.out.printf("query %s:\n", subject);
        Query query = QueryFactory.create(queryString);
        if (debug) query.serialize(new IndentedWriter(System.out, true));

        QuerySolutionMap initialBinding = new QuerySolutionMap();
        for (int i = 0; i < binding.length; i += 2)
            initialBinding.add((String) binding[i], (RDFNode) binding[i + 1]);

        List<List<RDFNode>> result = new ArrayList<>();
        try (QueryExecution exec = QueryExecutionFactory.create(query, model, initialBinding)) {
            ResultSet rs = exec.execSelect();

            while (rs.hasNext()) {
                QuerySolution rb = rs.nextSolution();

                Iterator<String> names = rb.varNames();
                List<RDFNode> line = new ArrayList<>();
                while (names.hasNext()) {
                    RDFNode node = rb.get(names.next());
                    line.add(node);
                    if (debug) print(node);
                }

                if (debug) System.out.println();
                result.add(line);
            }

            if (debug) System.out.printf("%s count: %d\n\n", subject, result.size());
        }

        return result;
    }

    public static List<RDFNode> execSingleQuery(String subject, String queryString, Object ... binding) {
        queryString = "PREFIX histo: <" + histoPrefix + "#>\n" +
                "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n" + queryString;

        if (debug) System.out.printf("query %s:\n", subject);
        Query query = QueryFactory.create(queryString);
        if (debug) query.serialize(new IndentedWriter(System.out, true));

        QuerySolutionMap initialBinding = new QuerySolutionMap();
        for (int i = 0; i < binding.length; i += 2)
            initialBinding.add((String) binding[i], (RDFNode) binding[i + 1]);

        List<RDFNode> result = new ArrayList<>();
        try (QueryExecution exec = QueryExecutionFactory.create(query, model, initialBinding)) {
            ResultSet rs = exec.execSelect();

            while (rs.hasNext()) {
                QuerySolution rb = rs.nextSolution();

                Iterator<String> names = rb.varNames();
                while (names.hasNext()) {
                    RDFNode node = rb.get(names.next());
                    result.add(node);
                    if (debug) print(node);
                }
            }

            if (debug) System.out.printf("\n%s count: %d\n\n", subject, result.size());
        }

        return result;
    }

    public static void print(Model model) {
        StmtIterator iter = model.listStatements();

        while (iter.hasNext()) {
            Statement stmt      = iter.nextStatement();  // get next statement
            Resource  subject   = stmt.getSubject();     // get the subject
            Property  predicate = stmt.getPredicate();   // get the predicate
            RDFNode   object    = stmt.getObject();      // get the object

            System.out.print(subject.toString());
            System.out.print(" " + predicate.toString() + " ");
            if (object instanceof Resource) {
                System.out.print(object);
            } else {
                System.out.print(" \"" + object.toString() + "\"");
            }

            System.out.println(" .");
        }
    }

    public static void print(RDFNode node) {
        String name = node.toString();
        int index = name.indexOf('#');
        if (index >= 0)
            name = name.substring(index + 1);
        System.out.printf("%s ", name);
    }

    public static String join(String ... args) {
        return String.join(System.getProperty("line.separator"), args);
    }
}
