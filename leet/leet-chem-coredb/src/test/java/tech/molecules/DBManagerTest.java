package tech.molecules;

import com.actelion.research.chem.StereoMolecule;
import tech.molecules.chem.coredb.*;
import tech.molecules.chem.coredb.sql.DBAssayResult;
import tech.molecules.chem.coredb.sql.DataValueImpl;
import tech.molecules.chem.coredb.sql.DBManager;
import tech.molecules.chem.coredb.sql.SQLHelper;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.*;

public class DBManagerTest {

    private static final String PROJECT_NAME = "ProtA Blocker";
    private static final String[] ASSAY_NAMES = {"IC50_ProtA", "HLM", "Solubility"};
    private static final int NUMBER_OF_COMPOUNDS = 200;
    private static final int BATCHES_PER_COMPOUND = 2;
    private static final double MEASUREMENT_VARIABILITY = 0.2;
    private static final Random random = new Random();

    public static void main(String[] args) {
        long ts_a = System.currentTimeMillis();
        test_A("sqlite");
        long ts_b = System.currentTimeMillis();
        test_A("h2db_mem");
        long ts_c = System.currentTimeMillis();
        System.out.println("sqlite:   "+(ts_b-ts_a));
        System.out.println("h2db_mem: "+(ts_c-ts_b));
    }

    public static void test_A(String db) {
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
        try (Connection connection = DriverManager.getConnection(dbUrl)) {
            DBManager dbManager = new DBManager(connection,sqlhelper);

            if(db.equals("sqlite")) {
                dbManager.createDatabaseSchema_sqlite();
            }
            else if(db.contains("h2db")) {
                dbManager.createDatabaseSchema_h2db();
            }

            Project project = dbManager.createProject(PROJECT_NAME,PROJECT_NAME);
            System.out.println("Created project: " + project.getName());

            DataType numericDataType = dbManager.createDataType("numeric");

            List<Assay> assays = new ArrayList<>();
            for (String assayName : ASSAY_NAMES) {
                Assay assay = dbManager.createAssay(assayName,project);
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
                        DataValue dataValue = new DataValueImpl(value,""+value);

                        dbManager.addDataValue(assayResult, parameter, dataValue);
                    }
                }
            }

            System.out.println("Inserted toy data.");

            // Test search
            testQueries(dbManager);
            //AssayResultQuery2 query = new AssayResultQuery2();
            //query.setProjectId(project.getId());
            //query.setAssay

        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    private static void testQueries(DBManager dbManager) throws SQLException {
        System.out.println("Testing queries...");

        System.out.println("Searching for assays with name 'IC50_ProtA'");
        List<Assay> assays = new ArrayList<>();//dbManager.searchAssays(new AssayQuery("IC50_ProtA", null));
        for (Assay assay : assays) {
            System.out.println("Found assay: " + assay.getId() + " - " + assay.getName());
        }

        System.out.println("Searching for tubes with batch ID 'Batch_1'");
        List<Tube> tubes = new ArrayList<>();//dbManager.searchTubes(new TubeQuery("Batch_1", null));
        for (Tube tube : tubes) {
            System.out.println("Found tube: " + tube.getId() + " - " + tube.getBatch().getId());
        }

        System.out.println("Searching for assay results with assay ID 1 and compound IDs 'C1', 'C2'");
        List<AssayResult> assayResults = DBAssayResult.searchAssayResults2(dbManager.getConnection(), new AssayResultQuery2(1,null,null, Arrays.asList("C0", "C1")));
        for (AssayResult result : assayResults) {
            System.out.println("Found assay result: " + result.getId() + " - Assay: " + result.getAssay().getId() + " - Tube: " + result.getTube().getId());
        }

        System.out.println("Searching for projects with name 'ProtA Blocker'");
        List<Project> projects = dbManager.searchProjects(new ProjectQuery("ProtA Blocker"));
        for (Project project : projects) {
            System.out.println("Found project: " + project.getId() + " - " + project.getName());
        }
    }



}