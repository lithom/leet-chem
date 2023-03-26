package tech.molecules.chem.coredb.sql;

import tech.molecules.chem.coredb.Assay;
import tech.molecules.chem.coredb.AssayParameter;
import tech.molecules.chem.coredb.Project;

import java.util.List;

public class AssayImpl implements Assay {
    private int id;
    private String name;
    private Project project;
    private List<AssayParameter> parameters;

    public AssayImpl(int id, String name, Project project, List<AssayParameter> parameters) {
        this.id = id;
        this.name = name;
        this.project = project;
        this.parameters = parameters;
    }

    public String getName() {
        return name;
    }

    public int getId() {
        return id;
    }

    public Project getProject() {return this.project;}

    public List<AssayParameter> getParameter() {
        return parameters;
    }
}
