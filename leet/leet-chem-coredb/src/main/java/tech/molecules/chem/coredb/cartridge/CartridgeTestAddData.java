package tech.molecules.chem.coredb.cartridge;

import com.actelion.research.chem.io.DWARFileParser;
import tech.molecules.leet.chem.ChemUtils;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CartridgeTestAddData {
    private static final String URL = "jdbc:postgresql://localhost:5434/coredb";
    private static final String USER = "coredb";
    private static final String PASSWORD = "coredb";

    public static void main(String[] args) {
        addPubchemStructures(1000000);
        //addWikipediaStructures();
    }

    public static void addPubchemStructures(long n) {
        List<String> structures = new ArrayList<>();
        String smilesfile = "/home/liphath1/Downloads/CID-SMILES";
        try(BufferedReader in = new BufferedReader(new FileReader(smilesfile))) {
            String line = null;
            while( (line=in.readLine()) != null && structures.size() < n) {
                String[] splits = line.split("\\t");
                try {
                    String idci = ChemUtils.parseSmiles(splits[1]).getIDCode();
                    structures.add(idci);
                }
                catch(Exception ex) {
                    ex.printStackTrace();
                }
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        insertStructures(structures);
    }

    public static void addWikipediaStructures() {
        String dwarfile = "/home/liphath1/datasets/Wikipedia_Compounds.dwar";
        DWARFileParser parser = new DWARFileParser(dwarfile);

        List<String> structures = new ArrayList<>();
        while(parser.next()) {
            structures.add( parser.getSpecialFieldData(parser.getSpecialFieldIndex("Structure")));
        }

        insertStructures(structures);
    }

    private static void insertStructures(List<String> structures) {
        String sql = "INSERT INTO your_table (value, structure) VALUES (?, ?)";

        try (Connection conn = DriverManager.getConnection(URL, USER, PASSWORD);
             PreparedStatement pstmt = conn.prepareStatement(sql)) {

            // Assuming you want to set the same value for each 'value' column
            int value = 123;

            for (String structure : structures) {
                pstmt.setInt(1, value);
                pstmt.setString(2, structure);
                pstmt.executeUpdate();
            }

            System.out.println("Insertion complete.");

        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}
