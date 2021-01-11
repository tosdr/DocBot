import { Regex } from '../models';

module.exports = {
    expression: new RegExp("^(((?=.*automatically)(?=.*renew))|((?=.*monthly)|(?=.*annually))((?=.*re-occurring)|(?=.*reoccur))|((?=.*recurring)(?=.*subscription)))", "mi"),
	expressionDont: new RegExp("", "i"),
	caseID: 384,
	name: "You authorise the service to charge a credit card supplied on re-occurring basis"
} as Regex;