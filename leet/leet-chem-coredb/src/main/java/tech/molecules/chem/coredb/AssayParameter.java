package tech.molecules.chem.coredb;

public interface AssayParameter {

    public int getId();
    public Assay getAssay();
    public DataType getDataType();
    public String getName();

}
