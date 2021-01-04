import { Regex } from '../models';

module.exports = {
    expression: new RegExp("^((?=.*automatically)(?=.*renew))|((?=.*monthly)|(?=.*annually))((?=.*re-occuring)|(reoccur))|(?=.*recurring)(?=.*subscription)", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 384,
	name: "You authorise the service to charge a credit card supplied on re-occurring basis"
} as Regex;