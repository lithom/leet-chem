package tech.molecules.leet.table.gui;

import javax.swing.*;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.function.Consumer;
import java.util.function.Function;

public class JActivatableTextField extends JPanel {
        private JCheckBox jba;
        private JLabel jla;
        private JTextField jtf;
        private Function<String,Boolean> validator;
        Consumer callback;

        public JActivatableTextField(String name, boolean activated, String text, int columns, Function<String,Boolean> validator, Consumer callback) {
            this.validator = validator;
            this.callback = callback;

            if(this.validator==null){this.validator = (s) -> true;}
            if(this.callback==null){this.callback = (s) -> {};}

            this.setLayout(new FlowLayout());
            this.jba = new JCheckBox();
            this.jla = new JLabel(name+" ");
            this.jtf = new JTextField(columns);
            this.jtf.setText(text);
            this.add(jba);
            this.add(jla);
            this.add(jtf);

            this.jba.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    jtf.setEnabled(jba.isSelected());
                    callback.accept(null);
                }
            });

            jtf.setEnabled(activated);
            this.jba.setSelected(activated);

            jtf.getDocument().addDocumentListener(new DocumentListener() {
                @Override
                public void insertUpdate(DocumentEvent e) {
                    checkValidity();
                    callback.accept(null);
                }
                @Override
                public void removeUpdate(DocumentEvent e) {
                    checkValidity();
                    callback.accept(null);
                }
                @Override
                public void changedUpdate(DocumentEvent e) {
                    checkValidity();
                    callback.accept(null);
                }
            });
        }

        private void checkValidity() {
            if(!this.validator.apply(jtf.getText())){ jtf.setBackground(Color.red.brighter()); }
            else{jtf.setBackground(UIManager.getColor("TextField.background"));}
        }

        public String getText() {return this.jtf.getText();}
        public boolean isActivated() {return this.jba.isSelected();}

}
