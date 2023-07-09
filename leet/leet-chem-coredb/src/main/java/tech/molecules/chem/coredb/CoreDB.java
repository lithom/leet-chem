package tech.molecules.chem.coredb;

import java.sql.Connection;
import java.sql.SQLException;
import java.util.List;
import java.util.Map;
import java.util.Set;

public interface CoreDB {

    public List<Assay> fetchAssays(Set<Integer> ids) throws Exception;

    public List<AssayResult> searchAssayResults(AssayResultQuery query) throws Exception;

    public List<Project> fetchProjects(Set<String> projectIds) throws SQLException;

    public List<Tube> searchTubes(TubeQuery query) throws Exception;

    public List<Project> searchProjects(ProjectQuery query) throws Exception;

    public List<Assay> searchAssays(AssayQuery query) throws Exception;

    public int getNumberOfMeasurements(Assay assay) throws Exception;

    public Map<String,Compound> fetchCompounds(List<String> identifiers);

    public List<Batch> fetchBatches(List<String> identifiers);

    public List<Tube> fetchTubes(List<String> tubeIds);

}
