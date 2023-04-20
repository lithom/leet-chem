package tech.molecules.chem.coredb;

public class AssayQuery {
    private String name;
    private String projectId;

    public AssayQuery() {

    }

    public AssayQuery(String name, String projectId) {
        this.name = name;
        this.projectId = projectId;
    }

    public String getName() {
        return name;
    }

    public String getProjectId() {
        return projectId;
    }
}