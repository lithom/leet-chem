package tech.molecules.leet.gui.chem.editor.sar;

// ... other imports ...
import com.actelion.research.chem.StereoMolecule;
import tech.molecules.leet.chem.sar.MultiFragment;
import tech.molecules.leet.chem.sar.SARElement;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;

public class MultiFragmentPanel extends JPanel implements MultiFragment.MultiFragmentListener {

    private JTabbedPane tabbedPane;
    private MultiFragment model;
    private List<MultiFragment.MultiFragmentListener> listeners = new ArrayList<>();

    public MultiFragmentPanel(MultiFragment model) {
        this.model = model;
        this.model.addMultiFragmentElementListener(this);


        this.reinit();
    }

    private void reinit() {
        this.removeAll();
        setLayout(new BorderLayout());
        tabbedPane = new JTabbedPane();
        add(tabbedPane, BorderLayout.CENTER);

        for(SARElement ei : this.model.getElements()) {
            SARElementPanel spi = new SARElementPanel(ei);
            this.tabbedPane.add("Element",spi);
            spi.addSARElementListener(new SARElementPanel.SARElementListener() {
                @Override
                public void onEdit() {
                    //fireSARElementEdited(ei);
                    model.setFragmentEdited(ei);
                }

                @Override
                public void onRemove() {

                }
            });
        }

        JButton addFragmentButton = new JButton("+");
        addFragmentButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                model.addMultiFragmentElement(new SARElement(new StereoMolecule())); // your model should fire a property change event here
            }
        });

//        JButton removeFragmentButton = new JButton("Remove Fragment");
//        removeFragmentButton.addActionListener(new ActionListener() {
//            @Override
//            public void actionPerformed(ActionEvent e) {
//                int selectedTabIndex = tabbedPane.getSelectedIndex();
//                model.removeMultiFragmentElement(selectedTabIndex); // your model should fire a property change event here
//            }
//        });

        JPanel topPanel = new JPanel();
        topPanel.setLayout(new BorderLayout());
        JTextField jtf  = new JTextField();
        topPanel.add(jtf,BorderLayout.CENTER);
        topPanel.add(addFragmentButton,BorderLayout.EAST);
        //buttonPanel.add(removeFragmentButton);
        add(topPanel, BorderLayout.NORTH);
        SwingUtilities.updateComponentTreeUI(this);
    }

    // Invoked when a MultiFragmentElement is added
    @Override
    public void onMultiFragmentElementAdded(SARElement element) {
        // Add visual representation of MultiFragmentElement to the appropriate tab
        this.reinit();
    }

    // Invoked when a MultiFragmentElement is removed
    @Override
    public void onMultiFragmentElementRemoved(int index) {
        this.reinit();
    }

    @Override
    public void onSARElementSelected(SARElement si) {

    }
}

