package tech.molecules.leet.chem.dataimport;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.sql.*;

public class ChemblActivitiesImport {

    public static void main(String[] args) {
        String url = "jdbc:postgresql://localhost/chembl_30";
        String username = "postgres";
        String password = "a";
        int min_num_measurements = 40;

        String query =  "SELECT\n" +
                        "    t.pref_name AS target_name,\n" +
                        "    t.tid AS target_id,\n" +
                        "    act.molregno AS structure_id,\n" +
                        "    cs.canonical_smiles AS chemical_structure,\n" +
                        "    act.standard_value AS activity_data,\n" +
                        "    act.standard_units AS activity_units\n" +
                        "FROM\n" +
                        "    target_dictionary t\n" +
                        "    LEFT JOIN assays ass ON ass.tid = t.tid\n" +
                        "    LEFT JOIN activities act ON act.assay_id = ass.assay_id\n" +
                        "    LEFT JOIN compound_structures cs ON cs.molregno = act.molregno\n" +
                        "WHERE\n" +
                        "    t.tid IN (\n" +
                        "        SELECT\n" +
                        "            t.tid\n" +
                        "        FROM\n" +
                        "            target_dictionary t\n" +
                        "            JOIN assays ass ON ass.tid = t.tid\n" +
                        "            JOIN activities act ON act.assay_id = ass.assay_id\n" +
                        "        WHERE\n" +
                        "            (act.standard_units = 'nM' OR act.standard_units = 'nm') AND\n" +
                        "            t.target_type = 'SINGLE PROTEIN'\n" +
                        "        GROUP BY\n" +
                        "            t.tid\n" +
                        "        HAVING\n" +
                        "            COUNT(DISTINCT act.assay_id) >= " + min_num_measurements + "\n" +
                        "\t\t\t) AND\n" +
                        "    (act.standard_units = 'nM' OR act.standard_units = 'nm')    \n" +
                        "ORDER BY\n" +
                        "    t.pref_name,\n" +
                        "    act.molregno;";


        try (Connection conn = DriverManager.getConnection(url, username, password);
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(query);
             FileWriter fwriter = new FileWriter("output_chembl_activities.csv")) {
             BufferedWriter writer = new BufferedWriter(fwriter);

            // Write CSV header
            writer.append("Target Name,Target ID,Structure ID,Chemical Structure[smiles],Activity Data[numeric],Activity Units\n");

            // Iterate through the result set and write each row to the CSV file
            rs.setFetchSize(1000);
            while (rs.next()) {
                String targetName = rs.getString("target_name");
                String targetId = rs.getString("target_id");
                String structureId = rs.getString("structure_id");
                String chemicalStructure = rs.getString("chemical_structure");
                String activityData = rs.getString("activity_data");
                String activityUnits = rs.getString("activity_units");
                //String confidence = rs.getString("confidence");

                // Write row to CSV file
                writer.append(String.format("%s,%s,%s,%s,%s,%s\n",
                        targetName, targetId, structureId, chemicalStructure, activityData, activityUnits));
            }

            writer.flush();
            writer.close();
            System.out.println("CSV file has been generated successfully!");

        } catch (SQLException | IOException e) {
            e.printStackTrace();
        }
    }
}
