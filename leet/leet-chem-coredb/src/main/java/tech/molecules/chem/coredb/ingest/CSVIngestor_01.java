package tech.molecules.chem.coredb.ingest;

import com.actelion.research.chem.StereoMolecule;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;
import tech.molecules.chem.coredb.*;
import tech.molecules.chem.coredb.sql.DBAssay;
import tech.molecules.chem.coredb.sql.DBManager;
import tech.molecules.chem.coredb.sql.DBManagerHelper;
import tech.molecules.chem.coredb.sql.SQLHelper;
import tech.molecules.leet.chem.ChemUtils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.sql.Connection;
import java.sql.Date;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.time.LocalDateTime;
import java.util.*;

/**
 *
 * File format (activities):
 *
 * target_name,target_id,structure_id,chemical_structure[smiles],activity_data[numeric],activity_units
 *
 * we just:
 * 1. create one project for every target_name
 * 2. create one assay for every target_name (target_name_IC50) with one assay parameter (IC50)
 * 3. create one compound for every structure id
 * 4. create one batch and tube for every compound
 * 5. create one measurement datapoint for every line
 *
 *
 */
public class CSVIngestor_01 {


    public static DataType numericDataType = null;

    public static void ingestFile(DBManager coredb, String csvFile, int maxLines) {

        int lineCnt = 0;
        String line;
        String cvsSplitBy = ",";

        // we keep track of the target names..
        Map<String,Triple<Project,Assay,AssayParameter>> targetNamesToAssay = new HashMap<>();
        // we also keep track of compounds. Pair is BatchId/TubeId
        Map<String,Pair<Batch,Tube>> compoundToBatchAndTube= new HashMap<>();

        // create "numeric" datatype:
        try {
            numericDataType = coredb.createDataType("numeric");
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }


        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {

            // Skip the header line
            br.readLine();

            while ((line = br.readLine()) != null) {
                if(lineCnt>maxLines) {break;}
                String[] data = line.split(cvsSplitBy);

                String targetName = data[0];
                String targetId = data[1];
                String structureId = data[2];
                String chemicalStructure = data[3];
                double activityData = Double.NaN;
                try{ activityData = Double.parseDouble(data[4]); } catch (NumberFormatException e) {
                    System.out.println("[INFO] NaN");
                }
                String activityUnits = data[5];

                String coredb_id = null;
                AssayParameter ap = null;
                Tube ti = null;

                if(!targetNamesToAssay.containsKey(targetName)) {
                    Triple<Project,Assay,AssayParameter> cta = createTarget(coredb,targetName);
                    targetNamesToAssay.put(targetName,cta);
                }
                ap = targetNamesToAssay.get(targetName).getRight();
                if(!compoundToBatchAndTube.containsKey(structureId)) {
                    Triple<Compound,Batch,Tube> cta = createCompound(coredb,structureId, chemicalStructure);
                    if(cta==null) {
                        System.out.println("[INFO] skip line: "+line);
                        continue;
                    }
                    coredb_id = cta.getLeft().getId();
                    compoundToBatchAndTube.put(structureId,Pair.of(cta.getMiddle(),cta.getRight()));
                }
                else {
                    coredb_id = compoundToBatchAndTube.get(structureId).getLeft().getCompound().getId();
                }
                ti = compoundToBatchAndTube.get(structureId).getRight();
                createActivityMeasurement(coredb,ap,ti,activityData);

                lineCnt++;
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static Triple<Project,Assay, AssayParameter> createTarget(DBManager coredb, String target) {
        // This function will be called for every target name
        System.out.println("Creating target: " + target);
        try {
            Project proj = coredb.createProject(target,target);
            Assay assay  = coredb.createAssay(target+"_IC50",proj);
            AssayParameter param = coredb.createAssayParameter(assay,numericDataType,"IC50");
            return Triple.of(proj,assay,param);
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return null;
    }

    private static Triple<Compound,Batch,Tube> createCompound(DBManager coredb, String structureId, String smiles) {
        // This function will be called for every new structure ID
        System.out.println("Creating compound: " + structureId + ", SMILES: " + smiles);
        StereoMolecule mi = null;
        try {
            mi = ChemUtils.parseSmiles(smiles);
        }
        catch (Exception ex) {
            System.out.println("[INFO] smiles parsing failed: "+smiles);
            return null;
        }
        Compound ci = null;
        Batch bi = null;
        Tube ti = null;
        try {
            ci = coredb.createCompound("chembl_"+structureId,mi);
            bi    = coredb.createBatch("batch_"+structureId,ci);
            ti    = coredb.createTube("T_"+structureId,bi);
        } catch (SQLException e) {
            e.printStackTrace();
            return null;
        }
        return Triple.of(ci,bi,ti);
    }

    private static AssayResult createActivityMeasurement(DBManager coredb, AssayParameter ap, Tube ti , double measuredActivity) {
        // Implement your logic here
        // This function will be called for every line in the CSV file
        System.out.println("Creating activity measurement: AssayParameter: " + ap.getName() +
                ", Structure ID: " + ti.getBatch().getCompound().getId() + ", Measured Activity: " + measuredActivity);

        try {
            AssayResult ar = coredb.createAssayResult(ap.getAssay(), new Date(System.currentTimeMillis()), ti);
            coredb.addDataValue(ar, ap, new NumericDataValue(measuredActivity));
            return ar;
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return null;
    }



    public static void main(String args[]) {

        String db = "h2db_mem";//"h2db_mem";
        int rii   = 1;

        String dbUrl = "";
        DBManager dbManager = null;
        try {
            //Connection connection = null; //DriverManager.getConnection(dbUrl);
            //dbManager = new DBManager(connection, sqlhelper);
            if (db.equals("sqlite")) {
                dbUrl = "jdbc:sqlite:chemdb_test_" + rii + ".db";
                dbManager = DBManagerHelper.getSQLite(dbUrl);
            } else if (db.contains("h2db")) {
                dbUrl = "jdbc:h2:mem:";
                dbManager = DBManagerHelper.getH2(dbUrl);
            } else if (db.equals("postgres")) {
                dbUrl = "jdbc:postgresql://localhost:5432/leet_chem_01";
                dbManager = DBManagerHelper.getPostgres(dbUrl,"postgres","a");
            }
            //connection = dbManager.getConnection();
            dbManager.createDatabaseSchema();
        } catch (SQLException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }

        ingestFile(dbManager,"output_chembl_activities.csv",20000);

        try {
            List<Assay> assays = dbManager.searchAssays(new AssayQuery());
            System.out.println("assays:");
            for(Assay asi : assays) {
                System.out.println(String.format("%s %d",asi.getName(),dbManager.getNumberOfMeasurements(asi)));
            }
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }



}
