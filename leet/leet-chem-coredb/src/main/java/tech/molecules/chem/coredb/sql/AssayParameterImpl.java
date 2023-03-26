package tech.molecules.chem.coredb.sql;

import tech.molecules.chem.coredb.Assay;
import tech.molecules.chem.coredb.AssayParameter;
import tech.molecules.chem.coredb.DataType;

public class AssayParameterImpl implements AssayParameter {

    private int id;
    private Assay assay;
    private DataType datatype;
    private String name;

    public AssayParameterImpl(int id, Assay assay, DataType datatype, String name) {
        this.id = id;
        this.assay = assay;
        this.datatype = datatype;
        this.name = name;
    }

    @Override
    public int getId() {
        return id;
    }

    @Override
    public Assay getAssay() {
        return assay;
    }

    @Override
    public DataType getDataType() {
        return datatype;
    }

    @Override
    public String getName() {
        return name;
    }
}
