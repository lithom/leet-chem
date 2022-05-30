package tech.molecules.leet.chem.dataimport;

import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.SmilesParser;
import com.actelion.research.chem.StereoMolecule;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * NOTES: About chembl structure:
 *
 * Get all activity assays with counts:
 *
 * SELECT public.activities.assay_id , count(public.activities.assay_id) from public.activities
 * 				join public.assays
 * 				on public.activities.assay_id = public.assays.assay_id
 * 				group by public.activities.assay_id
 * 				order by count(public.activities.assay_id) DESC
 *
 *
 * This is the sql that we take as input:
 *
 * SELECT public.compound_structures.molregno , public.compound_structures.canonical_smiles , public.target_dictionary.pref_name , public.assays.assay_id , public.activities.value , public.activities.units , public.products.trade_name , public.products.approval_date FROM public.compound_structures
 * join public.activities on public.activities.molregno = public.compound_structures.molregno
 * join public.assays on public.activities.assay_id = public.assays.assay_id
 * join public.target_dictionary on public.assays.tid = public.target_dictionary.tid
 * left outer join public.formulations on public.activities.molregno = public.formulations.molregno
 * left outer join public.products on public.formulations.product_id = public.products.product_id
 * where public.assays.assay_type = 'B' AND ( public.activities.units='uM' OR public.activities.units='um' OR public.activities.units='nm' )
 *
 *
 *
 */


public class ChemblPostgresImport {

    public static void main(String args[]) {

        String url    = "jdbc:postgresql://localhost/chembl_30";
        String user   = "postgres";
        //String port   = "5432";
        String pw     = "a";
        if(args.length>0) {
            url  = args[0];
            user = args[1];
            //port = args[2];
            pw   = args[2];
            //dbname = args[4];
        }

        Connection conn = null;
        try {
            conn = DriverManager.getConnection(url, user, pw);
            System.out.println("Connected to the PostgreSQL server successfully.");
        } catch (SQLException e) {
            System.out.println(e.getMessage());
        }

        System.out.println("mkay");


        try {
            createChemblDatafiles(conn,"C:\\Temp\\leet",50,1000);
        } catch (SQLException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }

    public static void createChemblDatafiles(Connection conn, String path_out, int max_hac, double max_ic50_nm) throws SQLException, IOException {

        String sql_a = "SELECT public.compound_structures.molregno , public.compound_structures.canonical_smiles , public.target_dictionary.pref_name , public.activities.standard_type , public.activities.value , public.activities.units , public.products.trade_name , public.products.approval_date FROM public.compound_structures\n" +
                "join public.activities on public.activities.molregno = public.compound_structures.molregno\n" +
                "join public.assays on public.activities.assay_id = public.assays.assay_id\n" +
                "join public.target_dictionary on public.assays.tid = public.target_dictionary.tid\n" +
                "left outer join public.formulations on public.activities.molregno = public.formulations.molregno\n" +
                "left outer join public.products on public.formulations.product_id = public.products.product_id\n" +
                "where public.assays.assay_type = 'B' AND ( public.activities.units='uM' OR public.activities.units='um' OR public.activities.units='nm' )\n";
        ResultSet rs = conn.createStatement().executeQuery(sql_a);

        rs.setFetchSize(1000);

        Map<String,ShortStructureEntry> short_entries = new HashMap<>();

        try(BufferedWriter out = new BufferedWriter(new FileWriter( path_out+ File.separator+"chembl_data.csv" ))) {

            List<String> header = new ArrayList<>();
            header.add("molregno");
            header.add("Structure[idcode]");
            header.add("CanonicSmiles");
            header.add("target");
            header.add("standard_type");
            header.add("activity");
            header.add("unit");
            header.add("trade_name");
            header.add("approval");
            header.add("hac");

            out.write(String.join(",", header) + "\n");

            while (rs.next()) {
                List<String> row_i = new ArrayList<>();

                // first process unit / activity:
                String unit_i = rs.getString(6);
                double val_i = rs.getDouble(5);

                if (unit_i.equals("uM") || unit_i.equals("um")) {
                    val_i = val_i * 1000;
                    unit_i = "nM";
                }
                if (!unit_i.equals("nM")) {
                    continue;
                }

                if(val_i > max_ic50_nm) {
                    continue;
                }

                String canonicSmiles = rs.getString(2);
                SmilesParser sp = new SmilesParser(SmilesParser.SMARTS_MODE_IS_SMILES, false);
                StereoMolecule mi = new StereoMolecule();

                try {
                    sp.parse(mi, canonicSmiles);
                } catch (Exception e) {
                    System.out.println("[ERROR] could not parse smiles");
                    continue;
                }

                mi.ensureHelperArrays(Molecule.cHelperCIP);

                if(mi.getAtoms()>max_hac) {
                    continue;
                }

                short_entries.put(mi.getIDCode(),new ShortStructureEntry(rs.getString(1),mi.getIDCode()));

                row_i.add(rs.getString(1));
                row_i.add(mi.getIDCode());
                //row_i.add("\""+canonicSmiles+"\"");
                row_i.add("");
                row_i.add(sanitizeCSVEntry(rs.getString(3)));
                row_i.add(sanitizeCSVEntry(rs.getString(4)));
                row_i.add(rs.getString(5));
                row_i.add(unit_i);
                row_i.add(sanitizeCSVEntry("trade_name:"+rs.getString(7)));
                row_i.add(sanitizeCSVEntry("Date:"+rs.getString(8)));

                row_i.add(""+mi.getAtoms());

                out.write(String.join(",", row_i) + "\n");
            }
            out.flush();
        }

        try(BufferedWriter out = new BufferedWriter(new FileWriter( path_out+ File.separator+"chembl_structures_short.csv" ))) {
            out.write("molregno,Structure[idcode]\n");
            for(String si : short_entries.keySet()) {
                ShortStructureEntry sse = short_entries.get(si);
                out.write(sse.molregno+","+sse.idc+"\n");
            }
            out.flush();
        }

    }

    public static String sanitizeCSVEntry(String str) {
        if(str==null){return "";}
        return str.replaceAll(",","_");
    }


    public static class ShortStructureEntry {
        public final String molregno;
        public final String idc;

        public ShortStructureEntry(String molregno, String idc) {
            this.molregno=molregno;
            this.idc=idc;
        }
    }

}
