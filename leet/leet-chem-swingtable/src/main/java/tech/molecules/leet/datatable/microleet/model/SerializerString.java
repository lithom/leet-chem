package tech.molecules.leet.datatable.microleet.model;

public class SerializerString implements MicroLeetDataValueSerializer<String> {

    @Override
    public String initFromString(String data) {
        return data;
    }

    @Override
    public String serializeToString(String val) {
        if(val==null) {return "";}
        return val;
    }

    @Override
    public Class getRepresentationClass() {
        return String.class;
    }
}
