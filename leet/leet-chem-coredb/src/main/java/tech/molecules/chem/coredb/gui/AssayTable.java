package tech.molecules.chem.coredb.gui;

import tech.molecules.chem.coredb.sql.DBManager;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.sql.SQLException;

public class AssayTable extends JPanel {
    private DBManager dbManager;
    private JTable table;
    private AssayTableModel tableModel;
    private JTextField filterTextField;

    public AssayTable(DBManager dbManager) {
        this.dbManager = dbManager;
        setLayout(new BorderLayout());

        initComponents();

        try {
            tableModel.loadData();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    private void initComponents() {
        tableModel = new AssayTableModel(dbManager);
        table = new JTable(tableModel);
        JScrollPane scrollPane = new JScrollPane(table);
        add(scrollPane, BorderLayout.CENTER);

        filterTextField = new JTextField();
        filterTextField.addKeyListener(new KeyListener() {
            @Override
            public void keyTyped(KeyEvent e) {
            }

            @Override
            public void keyPressed(KeyEvent e) {
            }

            @Override
            public void keyReleased(KeyEvent e) {
                tableModel.filterData(filterTextField.getText());
            }
        });
        add(filterTextField, BorderLayout.NORTH);
    }
}
