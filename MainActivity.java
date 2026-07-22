package com.example.calculator;

import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.example.calculator.engine.CalculatorEngine;
import com.example.calculator.engine.CalculatorState;

/**
 * MainActivity chỉ chịu trách nhiệm:
 *
 * - Bind Android UI
 * - Nhận click từ button
 * - Gửi action sang CalculatorEngine
 * - Đọc CalculatorState
 * - Render state lên màn hình
 *
 * Logic tính toán không đặt trong Activity.
 */
public class MainActivity extends AppCompatActivity {

    // =========================================================
    // ENGINE
    // =========================================================

    private CalculatorEngine calculatorEngine;


    // =========================================================
    // DISPLAY
    // =========================================================

    private TextView tvExpression;
    private TextView tvResult;


    // =========================================================
    // LIFECYCLE
    // =========================================================

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        EdgeToEdge.enable(this);

        setContentView(R.layout.activity_main);

        setupSystemBars();

        /*
         * Khởi tạo business logic.
         */
        calculatorEngine = new CalculatorEngine();

        /*
         * Bind các View trong XML.
         */
        bindViews();

        /*
         * Gắn interaction cho calculator.
         */
        setupDigitButtons();
        setupOperatorButtons();
        setupFunctionButtons();

        /*
         * Render state ban đầu:
         *
         * expression = ""
         * result = "0"
         */
        renderState();
    }


    // =========================================================
    // SYSTEM BARS
    // =========================================================

    /**
     * Giữ cách xử lý Edge-to-Edge của Empty Views Activity.
     */
    private void setupSystemBars() {

        ViewCompat.setOnApplyWindowInsetsListener(
                findViewById(R.id.main),
                (view, insets) -> {

                    Insets systemBars =
                            insets.getInsets(
                                    WindowInsetsCompat.Type.systemBars()
                            );

                    view.setPadding(
                            systemBars.left,
                            systemBars.top,
                            systemBars.right,
                            systemBars.bottom
                    );

                    return insets;
                }
        );
    }


    // =========================================================
    // BIND VIEW
    // =========================================================

    private void bindViews() {

        tvExpression =
                findViewById(
                        R.id.tvExpression
                );

        tvResult =
                findViewById(
                        R.id.tvResult
                );
    }


    // =========================================================
    // DIGIT BUTTONS
    // =========================================================

    private void setupDigitButtons() {

        bindDigitButton(
                R.id.btn0,
                "0"
        );

        bindDigitButton(
                R.id.btn1,
                "1"
        );

        bindDigitButton(
                R.id.btn2,
                "2"
        );

        bindDigitButton(
                R.id.btn3,
                "3"
        );

        bindDigitButton(
                R.id.btn4,
                "4"
        );

        bindDigitButton(
                R.id.btn5,
                "5"
        );

        bindDigitButton(
                R.id.btn6,
                "6"
        );

        bindDigitButton(
                R.id.btn7,
                "7"
        );

        bindDigitButton(
                R.id.btn8,
                "8"
        );

        bindDigitButton(
                R.id.btn9,
                "9"
        );
    }


    /**
     * Helper để tránh viết lại cùng một OnClickListener
     * cho 10 button số.
     */
    private void bindDigitButton(
            int buttonId,
            String digit
    ) {

        Button button =
                findViewById(buttonId);

        button.setOnClickListener(view -> {

            calculatorEngine.inputDigit(digit);

            renderState();
        });
    }


    // =========================================================
    // OPERATOR BUTTONS
    // =========================================================

    private void setupOperatorButtons() {

        bindOperatorButton(
                R.id.btnAdd,
                "+"
        );

        bindOperatorButton(
                R.id.btnSubtract,
                "−"
        );

        bindOperatorButton(
                R.id.btnMultiply,
                "×"
        );

        bindOperatorButton(
                R.id.btnDivide,
                "÷"
        );
    }


    /**
     * Helper chung cho:
     *
     * +
     * −
     * ×
     * ÷
     */
    private void bindOperatorButton(
            int buttonId,
            String operator
    ) {

        Button button =
                findViewById(buttonId);

        button.setOnClickListener(view -> {

            calculatorEngine.inputOperator(operator);

            renderState();
        });
    }


    // =========================================================
    // FUNCTION BUTTONS
    // =========================================================

    private void setupFunctionButtons() {

        /*
         * Decimal
         */
        findViewById(
                R.id.btnDecimal
        ).setOnClickListener(view -> {

            calculatorEngine.inputDecimal();

            renderState();
        });


        /*
         * Equals
         */
        findViewById(
                R.id.btnEquals
        ).setOnClickListener(view -> {

            calculatorEngine.calculate();

            renderState();
        });


        /*
         * AC
         */
        findViewById(
                R.id.btnClear
        ).setOnClickListener(view -> {

            calculatorEngine.clear();

            renderState();
        });


        /*
         * Backspace
         */
        findViewById(
                R.id.btnBackspace
        ).setOnClickListener(view -> {

            calculatorEngine.backspace();

            renderState();
        });


        /*
         * +/-
         */
        findViewById(
                R.id.btnToggleSign
        ).setOnClickListener(view -> {

            calculatorEngine.toggleSign();

            renderState();
        });


        /*
         * %
         */
        findViewById(
                R.id.btnPercent
        ).setOnClickListener(view -> {

            calculatorEngine.percent();

            renderState();
        });


        /*
         * Hai button này chưa implement ở increment Basic:
         *
         * btnHistory
         * btnMode
         *
         * Sẽ được nối ở các increment sau.
         */
    }


    // =========================================================
    // RENDER
    // =========================================================

    /**
     * UI chỉ đọc CalculatorState rồi hiển thị.
     *
     * Không thực hiện phép tính tại đây.
     */
    private void renderState() {

        CalculatorState state =
                calculatorEngine.getState();


        String expression =
                state.getExpression();

        String result =
                state.getResult();


        /*
         * Expression có thể rỗng ở trạng thái ban đầu.
         */
        tvExpression.setText(
                expression == null
                        ? ""
                        : expression
        );


        /*
         * Result luôn có giá trị,
         * mặc định là "0".
         */
        tvResult.setText(
                result == null
                        ? "0"
                        : result
        );
    }
}