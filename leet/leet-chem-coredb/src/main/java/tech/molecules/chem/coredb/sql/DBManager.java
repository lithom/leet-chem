package tech.molecules.chem.coredb.sql;

import com.actelion.research.chem.StereoMolecule;
import tech.molecules.chem.coredb.*;
import tech.molecules.leet.chem.ChemUtils;

import java.sql.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class DBManager {

    private SQLHelper helper;
    private Connection connection;

    public DBManager(Connection connection, SQLHelper helper) {
        this.connection = connection;
        this.helper = helper;
    }

    public void createDatabaseSchema_sqlite() throws SQLException {

        try (Statement statement = connection.createStatement()) {
            statement.execute("CREATE TABLE IF NOT EXISTS data_type (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT)");
            statement.execute("CREATE TABLE IF NOT EXISTS project (id TEXT PRIMARY KEY, name TEXT)");
            statement.execute("CREATE TABLE IF NOT EXISTS assay (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, project_id TEXT, FOREIGN KEY(project_id) REFERENCES project(id))");
            statement.execute("CREATE TABLE IF NOT EXISTS assay_parameter (id INTEGER PRIMARY KEY AUTOINCREMENT, assay_id INTEGER, data_type_id INTEGER, name TEXT, FOREIGN KEY(assay_id) REFERENCES assay(id), FOREIGN KEY(data_type_id) REFERENCES data_type(id))");
            statement.execute("CREATE TABLE IF NOT EXISTS assay_result (id INTEGER PRIMARY KEY AUTOINCREMENT, assay_id INTEGER, date DATE, tube_id TEXT, FOREIGN KEY(assay_id) REFERENCES assay(id), FOREIGN KEY(tube_id) REFERENCES tube(id))");
            statement.execute("CREATE TABLE IF NOT EXISTS batch (id TEXT PRIMARY KEY, compound_id TEXT, FOREIGN KEY(compound_id) REFERENCES compound(id))");
            statement.execute("CREATE TABLE IF NOT EXISTS compound (id TEXT PRIMARY KEY, idcode TEXT)");
            statement.execute("CREATE TABLE IF NOT EXISTS tube (id TEXT PRIMARY KEY, batch_id TEXT, FOREIGN KEY(batch_id) REFERENCES batch(id))");
            statement.execute("CREATE TABLE IF NOT EXISTS assay_result_data (assay_result_id INTEGER, assay_parameter_id INTEGER, double_value REAL, text_value TEXT, PRIMARY KEY (assay_result_id, assay_parameter_id), FOREIGN KEY(assay_result_id) REFERENCES assay_result(id), FOREIGN KEY(assay_parameter_id) REFERENCES assay_parameter(id))");
        }
    }

    public void createDatabaseSchema_h2db() throws SQLException {
        try (Statement statement = connection.createStatement()) {
            statement.execute("CREATE TABLE IF NOT EXISTS data_type (id INTEGER AUTO_INCREMENT PRIMARY KEY, name TEXT)");
            statement.execute("CREATE TABLE IF NOT EXISTS project (id TEXT PRIMARY KEY, name TEXT)");
            statement.execute("CREATE TABLE IF NOT EXISTS assay (id INTEGER AUTO_INCREMENT PRIMARY KEY, name TEXT, project_id TEXT, FOREIGN KEY(project_id) REFERENCES project(id))");
            statement.execute("CREATE TABLE IF NOT EXISTS assay_parameter (id INTEGER AUTO_INCREMENT PRIMARY KEY, assay_id INTEGER, data_type_id INTEGER, name TEXT, FOREIGN KEY(assay_id) REFERENCES assay(id), FOREIGN KEY(data_type_id) REFERENCES data_type(id))");
            statement.execute("CREATE TABLE IF NOT EXISTS compound (id TEXT PRIMARY KEY, idcode TEXT)");
            statement.execute("CREATE TABLE IF NOT EXISTS batch (id TEXT PRIMARY KEY, compound_id TEXT, FOREIGN KEY(compound_id) REFERENCES compound(id))");
            statement.execute("CREATE TABLE IF NOT EXISTS tube (id TEXT PRIMARY KEY, batch_id TEXT, FOREIGN KEY(batch_id) REFERENCES batch(id))");
            statement.execute("CREATE TABLE IF NOT EXISTS assay_result (id INTEGER AUTO_INCREMENT PRIMARY KEY, assay_id INTEGER, date DATE, tube_id TEXT, FOREIGN KEY(assay_id) REFERENCES assay(id), FOREIGN KEY(tube_id) REFERENCES tube(id))");
            statement.execute("CREATE TABLE IF NOT EXISTS assay_result_data (assay_result_id INTEGER, assay_parameter_id INTEGER, double_value REAL, text_value TEXT, PRIMARY KEY (assay_result_id, assay_parameter_id), FOREIGN KEY(assay_result_id) REFERENCES assay_result(id), FOREIGN KEY(assay_parameter_id) REFERENCES assay_parameter(id))");
        }
    }

    public Connection getConnection() {
        return this.connection;
    }

    public Project createProject(String id, String name) throws SQLException {
        //PreparedStatement statement = connection.prepareStatement("INSERT or IGNORE INTO project (id, name) VALUES (?, ?)");
        PreparedStatement statement = connection.prepareStatement(helper.getInsertOrIgnoreStatement("project","id, name","?, ?"));
        statement.setString(1, id);
        statement.setString(2, name);
        statement.executeUpdate();
        return new ProjectImpl(id, name);
    }

    public DataType createDataType(String name) throws SQLException {
        //String query = "INSERT INTO data_type (name) VALUES (?)";
        String query = helper.getInsertOrIgnoreStatement("data_type","name","?");
        PreparedStatement statement = connection.prepareStatement(query, Statement.RETURN_GENERATED_KEYS);
        statement.setString(1, name);

        statement.executeUpdate();
        ResultSet resultSet = statement.getGeneratedKeys();
        if (resultSet.next()) {
            int id = resultSet.getInt(1);
            return new DataTypeImpl(id, name);
        } else {
            throw new SQLException("Failed to create data type");
        }
    }

    public Assay createAssay(String name, Project project) throws SQLException {
        String query = "INSERT INTO assay (name, project_id) VALUES (?, ?)";
        PreparedStatement statement = connection.prepareStatement(query, Statement.RETURN_GENERATED_KEYS);
        statement.setString(1, name);
        statement.setString(2, project.getId());

        statement.executeUpdate();
        ResultSet resultSet = statement.getGeneratedKeys();
        if (resultSet.next()) {
            int id = resultSet.getInt(1);
            return new AssayImpl(id, name, project, new ArrayList<>() );
        } else {
            throw new SQLException("Failed to create assay");
        }
    }

    public AssayParameter createAssayParameter(Assay assay, DataType dataType, String name) throws SQLException {
        String query = "INSERT INTO assay_parameter (assay_id, data_type_id, name) VALUES (?, ?, ?)";
        PreparedStatement statement = connection.prepareStatement(query, Statement.RETURN_GENERATED_KEYS);
        statement.setInt(1, assay.getId());
        statement.setInt(2, dataType.getId());
        statement.setString(3, name);

        statement.executeUpdate();
        ResultSet resultSet = statement.getGeneratedKeys();


        if (resultSet.next()) {
            int id = resultSet.getInt(1);
            return new AssayParameterImpl(id, assay, dataType, name);
        } else {
            throw new SQLException("Failed to create assay parameter");
        }

    }

    public AssayResult createAssayResult(Assay assay, java.util.Date date, Tube tube) throws SQLException {
        String query = "INSERT INTO assay_result (assay_id, date, tube_id) VALUES (?, ?, ?)";
        PreparedStatement statement = connection.prepareStatement(query, Statement.RETURN_GENERATED_KEYS);
        statement.setInt(1, assay.getId());
        statement.setDate(2, new java.sql.Date(date.getTime()));
        statement.setString(3, tube.getId());

        statement.executeUpdate();
        ResultSet resultSet = statement.getGeneratedKeys();
        if (resultSet.next()) {
            long id = resultSet.getLong(1);
            return new AssayResultImpl(id, assay, date, tube, new HashMap<>());
        } else {
            throw new SQLException("Failed to create assay result");
        }
    }

    public Compound createCompound(String id, StereoMolecule molecule) throws SQLException {
        PreparedStatement statement = connection.prepareStatement("INSERT INTO compound (id) VALUES (?)");
        statement.setString(1, id);
        statement.executeUpdate();
        return new CompoundImpl(id, molecule);
    }

    public Batch createBatch(String id, Compound compound) throws SQLException {
        PreparedStatement statement = connection.prepareStatement("INSERT INTO batch (id, compound_id) VALUES (?, ?)");
        statement.setString(1, id);
        statement.setString(2, compound.getId());
        statement.executeUpdate();
        return new BatchImpl(id, compound);
    }

    public Tube createTube(String id, Batch batch) throws SQLException {
        PreparedStatement statement = connection.prepareStatement("INSERT INTO tube (id, batch_id) VALUES (?, ?)");
        statement.setString(1, id);
        statement.setString(2, batch.getId());
        statement.executeUpdate();
        return new TubeImpl(id, batch);
    }

    public void addDataValue(AssayResult assayResult, AssayParameter assayParameter, DataValue dataValue) throws SQLException {
        String query = "INSERT INTO assay_result_data (assay_result_id, assay_parameter_id, double_value, text_value) VALUES (?, ?, ?, ?)";
        PreparedStatement statement = connection.prepareStatement(query);
        statement.setLong(1, assayResult.getId());
        statement.setInt(2, assayParameter.getId());

        statement.setDouble(3, dataValue.getAsDouble());
        statement.setString(4, dataValue.getAsText());

        int affectedRows = statement.executeUpdate();
        if (affectedRows == 0) {
            throw new SQLException("Failed to add data value");
        }
    }




    public List<Tube> searchTubes(TubeQuery query) throws SQLException {
        List<Tube> result = new ArrayList<>();
        StringBuilder sb = new StringBuilder("SELECT tube.id as tube_id, tube.batch_id, batch.compound_id, compound.idcode ")
                .append("FROM tube ")
                .append("JOIN batch ON tube.batch_id = batch.id ")
                .append("JOIN compound ON batch.compound_id = compound.id ")
                .append("WHERE 1=1");

        if (query.getBatchId() != null) {
            sb.append(" AND tube.batch_id = ?");
        }

        PreparedStatement statement = connection.prepareStatement(sb.toString());
        int parameterIndex = 1;

        if (query.getBatchId() != null) {
            statement.setString(parameterIndex, query.getBatchId());
        }

        ResultSet resultSet = statement.executeQuery();
        while (resultSet.next()) {
            String tubeId = resultSet.getString("tube_id");
            String batchId = resultSet.getString("batch_id");
            String compoundId = resultSet.getString("compound_id");
            String compoundIdcode = resultSet.getString("idcode");

            StereoMolecule molecule = ChemUtils.parseIDCode(compoundIdcode);
            Compound compound = new CompoundImpl(compoundId, molecule);
            Batch batch = new BatchImpl(batchId, compound);

            result.add(new TubeImpl(tubeId, batch));
        }

        return result;
    }


    public List<Project> searchProjects(ProjectQuery query) throws SQLException {
        List<Project> result = new ArrayList<>();
        StringBuilder sb = new StringBuilder("SELECT * FROM project WHERE 1=1");

        if (query.getName() != null) {
            sb.append(" AND name LIKE ?");
        }

        PreparedStatement statement = connection.prepareStatement(sb.toString());
        int parameterIndex = 1;

        if (query.getName() != null) {
            statement.setString(parameterIndex, "%" + query.getName() + "%");
        }

        ResultSet resultSet = statement.executeQuery();
        while (resultSet.next()) {
            String id = resultSet.getString("id");
            String name = resultSet.getString("name");

            result.add(new ProjectImpl(id, name));
        }

        return result;
    }


}


