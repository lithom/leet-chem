package tech.molecules.leet.datatable.swing3;

import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import javax.swing.table.TableCellEditor;
import javax.swing.table.TableModel;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;

public class InteractiveSwingTable extends JTable {

    public InteractiveSwingTable(TableModel model) {
        super(model);
        initGlobalMouseListener();
    }

    private void initGlobalMouseListener() {
        MouseMotionAdapter globalMouseListener = new MouseMotionAdapter() {
            private int lastRow = -1;
            private int lastColumn = -1;

            @Override
            public void mouseMoved(MouseEvent e) {
                Point p = e.getPoint();
                int row = rowAtPoint(p);
                int column = columnAtPoint(p);

                if (row != lastRow || column != lastColumn) {
                    if (isEditing()) {
                        TableCellEditor cellEditor = getCellEditor();
                        if (cellEditor != null) {
                            cellEditor.cancelCellEditing();
                        }
                    }

                    lastRow = row;
                    lastColumn = column;
                    if (row != -1 && column != -1) {
                        editCellAt(row, column);
                    }
                }
            }
        };

        addMouseMotionListener(globalMouseListener);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("Extended JTable Example");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

            String[] columnNames = {"Column 1", "Column 2", "Column 3"};
            Object[][] data = {
                    {"A1", "B1", "C1"},
                    {"A2", "B2", "C2"},
                    {"A3", "B3", "C3"}
            };
            DefaultTableModel model = new DefaultTableModel(data, columnNames);
            InteractiveSwingTable table = new InteractiveSwingTable(model);

            JScrollPane scrollPane = new JScrollPane(table);
            frame.add(scrollPane, BorderLayout.CENTER);
            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }
}