package tech.molecules.chem.coredb;

import com.actelion.research.chem.StereoMolecule;

import java.sql.Connection;
import java.sql.SQLException;

public interface CoreDBWriter {

    //public void createDatabaseSchema_sqlite() throws Exception;

    //public void createDatabaseSchema_h2db() throws Exception;

    public void createDatabaseSchema() throws Exception;

    Connection getConnection();

    Project createProject(String id, String name) throws Exception;

    //DataType createDataType(String name) throws Exception;

    Assay createAssay(String name, Project project) throws Exception;

    AssayParameter createAssayParameter(Assay assay, DataType dataType, String name) throws Exception;

    AssayResult createAssayResult(Assay assay, java.util.Date date, Tube tube) throws Exception;

    Compound createCompound(String id, StereoMolecule molecule) throws Exception;

    Batch createBatch(String id, Compound compound) throws Exception;

    Tube createTube(String id, Batch batch) throws Exception;

    void addDataValue(AssayResult assayResult, AssayParameter assayParameter, DataValue dataValue) throws Exception;

}
