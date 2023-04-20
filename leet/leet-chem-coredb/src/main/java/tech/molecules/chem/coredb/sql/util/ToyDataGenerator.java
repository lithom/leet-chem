package tech.molecules.chem.coredb.sql.util;

import com.actelion.research.chem.StereoMolecule;
import tech.molecules.chem.coredb.*;
import tech.molecules.chem.coredb.sql.DBManager;
import tech.molecules.chem.coredb.sql.DataValueImpl;
import tech.molecules.chem.coredb.sql.SQLHelper;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Random;

public class ToyDataGenerator {

    private static final String PROJECT_NAME = "ProtA Blocker";
    private static final String[] ASSAY_NAMES = {"IC50_ProtA", "HLM", "Solubility"};
    private static final int NUMBER_OF_COMPOUNDS = 200;
    private static final int BATCHES_PER_COMPOUND = 2;
    private static final double MEASUREMENT_VARIABILITY = 0.2;
    private static final Random random = new Random();

    public static void main(String[] args) {
        long ts_a = System.currentTimeMillis();
        generateToyData_A("sqlite");
        long ts_b = System.currentTimeMillis();
        generateToyData_A("h2db_mem");
        long ts_c = System.currentTimeMillis();
        System.out.println("sqlite:   "+(ts_b-ts_a));
        System.out.println("h2db_mem: "+(ts_c-ts_b));




    }

    public static DBManager generateToyData_A(String db) {
        Random ri = new Random();
        int rii = Math.abs( ri.nextInt() ) % 10000;

        String dbUrl = "";
        SQLHelper sqlhelper = null;
        if(db.equals("sqlite")) {
            dbUrl = "jdbc:sqlite:chemdb_test_" + rii + ".db";
            sqlhelper = new SQLHelper.SqliteHelper();
        }
        else if(db.equals("h2db_mem")) {
            dbUrl = "jdbc:h2:mem:";
            sqlhelper = new SQLHelper.H2Helper();
        }

        DBManager dbManager = null;
        try {
            Connection connection = DriverManager.getConnection(dbUrl);
            dbManager = new DBManager(connection, sqlhelper);

            if (db.equals("sqlite")) {
                dbManager.createDatabaseSchema_sqlite();
            } else if (db.contains("h2db")) {
                dbManager.createDatabaseSchema_h2db();
            }

            Project project = dbManager.createProject(PROJECT_NAME, PROJECT_NAME);
            System.out.println("Created project: " + project.getName());

            DataType numericDataType = dbManager.createDataType("numeric");

            List<Assay> assays = new ArrayList<>();
            for (String assayName : ASSAY_NAMES) {
                Assay assay = dbManager.createAssay(assayName, project);
                AssayParameter parameter = dbManager.createAssayParameter(assay, numericDataType, "value");
                assays.add(assay);
                System.out.println("Created assay: " + assay.getName());
                assay.getParameter().add(parameter);
            }

            for (int i = 1; i <= NUMBER_OF_COMPOUNDS; i++) {
                String compoundId = "C" + i;
                StereoMolecule molecule = new StereoMolecule(); // Assuming you have a StereoMolecule constructor.
                Compound compound = dbManager.createCompound(compoundId, molecule);

                for (int j = 1; j <= BATCHES_PER_COMPOUND; j++) {
                    String batchId = compoundId + "_B" + j;
                    Batch batch = dbManager.createBatch(batchId, compound);

                    String tubeId = batchId + "_T";
                    Tube tube = dbManager.createTube(tubeId, batch);

                    for (Assay assay : assays) {
                        Date date = new Date(System.currentTimeMillis());

                        AssayResult assayResult = dbManager.createAssayResult(assay, date, tube);
                        AssayParameter parameter = assay.getParameter().get(0);

                        double value = 1 + MEASUREMENT_VARIABILITY * (random.nextDouble() * 2 - 1);
                        DataValue dataValue = new DataValueImpl(value, "" + value);

                        dbManager.addDataValue(assayResult, parameter, dataValue);
                    }
                }
            }

            System.out.println("Inserted toy data.");

            // Test search
            //testQueries(dbManager);
            //AssayResultQuery2 query = new AssayResultQuery2();
            //query.setProjectId(project.getId());
            //query.setAssay
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
        return dbManager;

    }

}
