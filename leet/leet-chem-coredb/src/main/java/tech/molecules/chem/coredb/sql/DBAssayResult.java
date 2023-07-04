package tech.molecules.chem.coredb.sql;

import tech.molecules.chem.coredb.*;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.*;

public class DBAssayResult {

    public static List<AssayResult> searchAssayResults(Connection connection, AssayResultQuery query) throws SQLException {
        StringBuilder queryBuilder = new StringBuilder(
                "SELECT  assay_result.id , assay_result.date , assay_result.assay_id , " +
                        " assay_result.tube_id AS tube_id , tube.batch_id AS batch_id , " +
                        "compound.id AS compound_id , compound.idcode AS compound_idcode " +
                        " FROM assay_result " +
                        "JOIN assay ON assay_result.assay_id = assay.id " +
                        "JOIN tube ON assay_result.tube_id = tube.id " +
                        "JOIN batch ON tube.batch_id = batch.id " +
                        "JOIN compound ON batch.compound_id = compound.id "
        );

        List<Object> parameters = new ArrayList<>();
        boolean hasCondition = false;

        if (query.getAssayId() != null) {
            queryBuilder.append(hasCondition ? " AND" : " WHERE");
            queryBuilder.append(" assay_result.assay_id = ?");
            parameters.add(query.getAssayId());
            hasCondition = true;
        }

        if (query.getFromDate() != null) {
            queryBuilder.append(hasCondition ? " AND" : " WHERE");
            queryBuilder.append(" assay_result.date >= ?");
            parameters.add(query.getFromDate());
            hasCondition = true;
        }

        if (query.getToDate() != null) {
            queryBuilder.append(hasCondition ? " AND" : " WHERE");
            queryBuilder.append(" assay_result.date <= ?");
            parameters.add(query.getToDate());
            hasCondition = true;
        }

        if (query.getCompoundIds() != null && !query.getCompoundIds().isEmpty()) {
            queryBuilder.append(hasCondition ? " AND" : " WHERE");
            queryBuilder.append(" compound.id IN (");
            StringJoiner compoundIdPlaceholders = new StringJoiner(", ");
            for (String compoundId : query.getCompoundIds()) {
                compoundIdPlaceholders.add("?");
                parameters.add(compoundId);
            }
            queryBuilder.append(compoundIdPlaceholders);
            queryBuilder.append(")");
        }

        PreparedStatement statement = connection.prepareStatement(queryBuilder.toString());

        for (int i = 0; i < parameters.size(); i++) {
            Object parameter = parameters.get(i);
            if (parameter instanceof Integer) {
                statement.setInt(i + 1, (Integer) parameter);
            } else if (parameter instanceof Date) {
                statement.setDate(i + 1, new java.sql.Date(((Date) parameter).getTime()));
            } else if (parameter instanceof String) {
                statement.setString(i + 1, (String) parameter);
            }
        }

        ResultSet resultSet = statement.executeQuery();
        resultSet.setFetchSize(10000);
        List<AssayResult> results = new ArrayList<>();

        Set<Integer> requiredAssays = new HashSet<>();
        Map<Long,Integer> mapAssayResultIdToAssayId = new HashMap<>();
        Map<Long,AssayResult> mapAssayResults = new HashMap<>();

        while (resultSet.next()) {
            long id = resultSet.getLong("id");
            Date date = resultSet.getDate("date");

            int assayId = resultSet.getInt("assay_id");
            //String assayName = resultSet.getString("assay.name");
            //Assay assay = new AssayImpl(assayId, assayName);
            Assay assay = null;
            requiredAssays.add(assayId);
            mapAssayResultIdToAssayId.put(id,assayId);


            String tubeId = resultSet.getString("tube_id");
            String batchId = resultSet.getString("batch_id");
            String compoundId = resultSet.getString("compound_id");
            String compoundIdcode = resultSet.getString("compound_idcode");
            Compound compound = new CompoundImpl(compoundId,compoundIdcode); // You need to implement this method to fetch the StereoMolecule based on the compound ID.
            Batch batch = new BatchImpl(batchId, compound);
            Tube tube = new TubeImpl(tubeId, batch);
            AssayResult ari = new AssayResultImpl(id, null, date, tube,null);
            results.add(ari);
            mapAssayResults.put(id,ari);
        }

        // fetch assays..
        List<Assay> fetched_assays = DBAssay.fetchAssays(connection,requiredAssays);
        Map<Integer,Assay> sorted_assays = new HashMap<>();
        fetched_assays.stream().forEach( xi -> sorted_assays.put(xi.getId(),xi) );

        // fetch data values..
        Map<Long, Map<AssayParameter, DataValue>> data_maps = fetchDataMaps(connection, mapAssayResultIdToAssayId , sorted_assays );
        for(AssayResult ari : results) {
            ((AssayResultImpl)ari).setDataValueMap(data_maps.get(ari.getId()));
        }

        // set assays..
        mapAssayResultIdToAssayId.entrySet().stream().forEach( xi -> ((AssayResultImpl) mapAssayResults.get( xi.getKey() )).setAssay( sorted_assays.get(xi.getValue()) ) );

        return results;
    }

    public static Map<Long, Map<AssayParameter, DataValue>> fetchDataMaps(Connection connection, Map<Long,Integer> assayResultIds, Map<Integer,Assay> assays_sorted) throws SQLException {
        if (assayResultIds == null || assayResultIds.isEmpty()) {
            return Collections.emptyMap();
        }

        StringBuilder queryBuilder = new StringBuilder(
                "SELECT assay_result_data.assay_result_id, " +
                        "assay_parameter.id AS parameter_id, assay_parameter.name AS parameter_name, " +
                        "assay_parameter.data_type AS data_type_name, " + "assay_parameter.assay_id AS assay_id, " +
                        "assay_result_data.double_value , assay_result_data.text_value AS text_value " +
                        "FROM assay_result_data " +
                        "JOIN assay_parameter ON assay_result_data.assay_parameter_id = assay_parameter.id " +
                        //"JOIN data_type ON assay_parameter.data_type_id = data_type.id " +
                        "WHERE assay_result_data.assay_result_id IN ("
        );

        StringJoiner idPlaceholders = new StringJoiner(", ");
        for (Long id : assayResultIds.keySet()) {
            idPlaceholders.add("?");
        }
        queryBuilder.append(idPlaceholders);
        queryBuilder.append(")");

        PreparedStatement statement = connection.prepareStatement(queryBuilder.toString());
        int i = 1;
        for (Long id : assayResultIds.keySet()) {
            statement.setLong(i++, id);
        }

        ResultSet resultSet = statement.executeQuery();
        Map<Long, Map<AssayParameter, DataValue>> dataMaps = new HashMap<>();
        while (resultSet.next()) {
            long assayResultId = resultSet.getLong("assay_result_id");
            int parameterId = resultSet.getInt("parameter_id");
            String parameterName = resultSet.getString("parameter_name");
            //String dataTypeName = resultSet.getString("data_type_name");
            String dataTypeId = resultSet.getString("data_type_name");
            //int assay_id = resultSet.getInt("assay_id");
            int assay_id = assayResultIds.get(assayResultId);
            //int data_type_id = resultSet.getInt("data_type_id");
            //Assay dummyAssay = new AssayImpl(0, "",null); // We create a dummy assay object as we only need the parameter.
            //DataType dataType = new DataTypeImpl(data_type_id, dataTypeName);
            DataType dataType = DataType.fromValue(dataTypeId);

            AssayParameter parameter = new AssayParameterImpl(parameterId, assays_sorted.get(assay_id) , dataType, parameterName);

            double doubleValue = resultSet.getDouble("double_value");
            String textValue = resultSet.getString("text_value");
            DataValue dataValue = new DataValueImpl(doubleValue,textValue);

            Map<AssayParameter, DataValue> dataMap = dataMaps.get(assayResultId);
            if (dataMap == null) {
                dataMap = new HashMap<>();
                dataMaps.put(assayResultId, dataMap);
            }
            dataMap.put(parameter, dataValue);
        }

        return dataMaps;
    }

}
