import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import freemarker.template.Configuration;
import freemarker.template.Template;
import freemarker.template.TemplateException;
import org.apache.jena.atlas.io.IndentedWriter;
import org.apache.jena.query.*;
import org.apache.jena.rdf.model.*;
import org.apache.jena.riot.RDFDataMgr;
import org.apache.jena.util.FileUtils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class Search {
    public static Model model = ModelFactory.createDefaultModel();
    public static String histoPrefix = "https://pathoml-1308125782.cos.ap-chengdu.myqcloud.com/PathoML.owl";
    private static final boolean debug = false, generate = false;

    public static void transform(String[] args) throws IOException, TemplateException {
        class CellProp {
            public final int grade, area;
            public final double major_axis, circularity, entropy;
            public CellProp(ArrayNode node) {
                grade = node.get(0).asInt();
                area = node.get(1).asInt();
                major_axis = node.get(2).asDouble();
                circularity = node.get(3).asDouble();
                entropy = node.get(4).asDouble();
            }
        }

        Template template = null;
        if (generate) {
            Configuration cfg = new Configuration(Configuration.VERSION_2_3_31);
            cfg.setDirectoryForTemplateLoading(new File("."));
            template = cfg.getTemplate("Template.xml");
        }

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode root = (ObjectNode) objectMapper.readTree(new File("CellData.json"));
        Map<String, List<CellProp>> cellPropMap = new HashMap<>();
        for (Iterator<Map.Entry<String, JsonNode>> iter = root.fields(); iter.hasNext(); ) {
            Map.Entry<String, JsonNode> entry = iter.next();
            ArrayNode slide = (ArrayNode) entry.getValue();
            List<CellProp> cellProps = new ArrayList<>();
            List<Integer> regions = new ArrayList<>();
            for (Iterator<JsonNode> iter2 = slide.elements(); iter2.hasNext(); ) {
                ArrayNode array = (ArrayNode) iter2.next();
                CellProp prop = new CellProp(array);
                cellProps.add(prop);
                regions.add(prop.grade);
            }
            cellPropMap.put(entry.getKey(), cellProps);

            if (generate) {
                Map<String, Object> dataModel = new HashMap<>();
                dataModel.put("id", entry.getKey());
                dataModel.put("regions", regions);
                template.process(dataModel, new FileWriter("Representation/" + entry.getKey() + ".xml"));
            }
        }

        String[] queries = new String[] {
                FileUtils.readWholeFileAsUTF8("Grade1.rq"),
                FileUtils.readWholeFileAsUTF8("Grade2.rq"),
                FileUtils.readWholeFileAsUTF8("Grade3.rq"),
                FileUtils.readWholeFileAsUTF8("Endothelial.rq")
        };

        int[] cnt = new int[4];
        double[] area = new double[4], major_axis = new double[4], circularity = new double[4], entropy = new double[4];
        for (Iterator<Map.Entry<String, JsonNode>> iter = root.fields(); iter.hasNext(); ) {
            Map.Entry<String, JsonNode> entry = iter.next();

            model.read(RDFDataMgr.open("Representation/" + entry.getKey() + ".xml"), null);
            List<CellProp> cellProps = cellPropMap.get(entry.getKey());
            for (List<RDFNode> data : execQuery("grade 1", queries[0])) {
                cnt[0]++;
                CellProp prop = cellProps.get(Integer.parseInt(data.get(0).toString().replaceAll(",", "")) - 1);
                area[0] += prop.area;
                major_axis[0] += prop.major_axis;
                circularity[0] += prop.circularity;
                entropy[0] += prop.entropy;
            }

            for (List<RDFNode> data : execQuery("grade 2", queries[1])) {
                cnt[1]++;
                CellProp prop = cellProps.get(Integer.parseInt(data.get(0).toString().replaceAll(",", "")) - 1);
                area[1] += prop.area;
                major_axis[1] += prop.major_axis;
                circularity[1] += prop.circularity;
                entropy[1] += prop.entropy;
            }

            for (List<RDFNode> data : execQuery("grade 3", queries[2])) {
                cnt[2]++;
                CellProp prop = cellProps.get(Integer.parseInt(data.get(0).toString().replaceAll(",", "")) - 1);
                area[2] += prop.area;
                major_axis[2] += prop.major_axis;
                circularity[2] += prop.circularity;
                entropy[2] += prop.entropy;
            }

            for (List<RDFNode> data : execQuery("endothelial", queries[3])) {
                cnt[3]++;
                CellProp prop = cellProps.get(Integer.parseInt(data.get(0).toString().replaceAll(",", "")) - 1);
                area[3] += prop.area;
                major_axis[3] += prop.major_axis;
                circularity[3] += prop.circularity;
                entropy[3] += prop.entropy;
            }

            model.removeAll();
        }

        for (String query : args) {
            double[] data = null;
            switch (query) {
                case "area": data = area; break;
                case "diameter": data = major_axis; break;
                case "circularity": data = circularity; break;
                case "entropy": data = entropy; break;
            }

            if (data != null) {
                System.out.printf("average %s:\n", query);
                if (cnt[0] > 0) System.out.printf("grade 1 (%d): %f\n", cnt[0], data[0] / cnt[0]);
                if (cnt[1] > 0) System.out.printf("grade 2 (%d): %f\n", cnt[1], data[1] / cnt[1]);
                if (cnt[2] > 0) System.out.printf("grade 3 (%d): %f\n", cnt[2], data[2] / cnt[2]);
                if (cnt[3] > 0) System.out.printf("endothelial (%d): %f\n", cnt[3], data[3] / cnt[3]);
                System.out.println();
            }
        }
    }

    public static void main(String[] args) throws Exception {
        transform(args);
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
