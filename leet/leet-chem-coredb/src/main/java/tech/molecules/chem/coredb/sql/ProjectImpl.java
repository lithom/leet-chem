package tech.molecules.chem.coredb.sql;

import tech.molecules.chem.coredb.Project;

public class ProjectImpl implements Project {
    private String id;
    private String name;

    public ProjectImpl(String id, String name) {
        this.id = id;
        this.name = name;
    }

    public String getId() { return id; }
    public String getName() { return name; }
}