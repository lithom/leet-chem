package tech.molecules.leet.datatable.microleet.model;

public class SerializerNumeric implements MicroLeetDataValueSerializer<Double> {

    @Override
    public Double initFromString(String data) {
        try {
            double v = Double.parseDouble(data);
            return v;
        }
        catch (Exception ex) {
            // ok..
        }
        return null;
    }

    @Override
    public String serializeToString(Double val) {
        if(val==null) {return "";}
        return ""+val;
    }

    @Override
    public Class getRepresentationClass() {
        return Double.class;
    }
}
