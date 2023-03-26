package tech.molecules.chem.coredb.sql;

import tech.molecules.chem.coredb.DataType;

public class DataTypeImpl implements DataType {

    private int id;
    private String name;

    public DataTypeImpl(int id, String datatypeName) {
        this.id = id;
        this.name = datatypeName;
    }

    @Override
    public int getId() {
        return id;
    }

    @Override
    public String getName() {
        return this.name;
    }

}
